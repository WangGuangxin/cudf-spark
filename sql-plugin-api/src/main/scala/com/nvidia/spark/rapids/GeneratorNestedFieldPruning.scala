/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import scala.collection.mutable

import org.apache.spark.sql.catalyst.expressions.{ArraysZip, Attribute, AttributeReference, Exists, ExprId, Expression, GetArrayStructFields, GetStructField, IsNotNull, IsNull, Size, SubqueryExpression}
import org.apache.spark.sql.catalyst.expressions.{Explode, PosExplode}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Generate, GlobalLimit, Join, LocalLimit, LogicalPlan, Project, Repartition, RepartitionByExpression, ResolvedHint, Sample, Sort, SubqueryAlias, Window}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ArrayType, StructType}

/**
 * Prunes unused struct fields from the input of `explode`/`posexplode` over an
 * `array<struct<...>>` when the generated struct is only ever read field by field.
 *
 * Spark's own `GeneratorNestedColumnAliasing` only prunes the generator input when exactly one
 * nested field of the generator output is referenced, and only when the generate sits directly
 * below a Project (a Filter in between, as in `WHERE l.f = ...`, already prevents it). With two
 * or more fields (for example
 * `explode(lineitem_data) AS l` followed by `l.l_receiptdate > l.l_commitdate`) the whole
 * array of structs is read from the file, decoded and carried through every operator up to the
 * point where the fields are extracted, which can include joins and shuffles.
 *
 * This rule rewrites
 * {{{
 *   Generate(explode(arr), ..., [g])  where g is only used as g.f1, g.f2, ...
 * }}}
 * to
 * {{{
 *   Generate(explode(arrays_zip(arr.f1, arr.f2, ...)), ..., [g'])
 * }}}
 * and remaps every `g.fi` to the matching field of `g'`. With a single field `f` it produces
 * `explode(arr.f)` instead and replaces `g.f` by the generated value. Because the generator input now only
 * contains `GetArrayStructFields` over `arr`, Spark's existing nested column aliasing pushes the
 * extractions down and `SchemaPruning` reads only those leaf columns. All arguments of the
 * `arrays_zip` are field projections of the same array, so they have the same length and the same
 * nullness as `arr`; explode therefore produces the same number of rows (also for `outer`), and
 * every field value is unchanged. The struct-level value of `g` is never observed because the
 * rule only fires when no expression uses `g` other than to extract a field (or to forward it
 * unchanged through a `Project`) and `g` is not part of a plan or subquery output.
 *
 * `size`, `isnotnull` and `isnull` over such a zip (which `InferFiltersFromGenerate` adds) are
 * rewritten to use the first zipped field, so the inferred filters do not reference the whole
 * array either and stay simple enough to run on the GPU.
 *
 * The rule runs once after analysis and again as an operator optimization rule. The post-analysis
 * run matters for EXISTS subqueries: Spark optimizes a subquery plan on its own (where its
 * `SELECT *` output makes every field look used) before the outer plan, and the scan inside it is
 * not pruned again after the subquery has been rewritten to a semi/anti join. From the outer plan
 * the output of an EXISTS subquery is known to be unused.
 *
 * Disabled together with Spark's nested schema pruning
 * (`spark.sql.optimizer.nestedSchemaPruning.enabled=false`), or individually by adding this
 * rule's name to `spark.sql.optimizer.excludedRules`.
 */
object GeneratorNestedFieldPruning extends Rule[LogicalPlan] {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!SQLConf.get.nestedSchemaPruningEnabled) {
      plan
    } else {
      var result = plan
      candidates(plan).foreach { out =>
        result = pruneOne(result, out)
      }
      simplifyZipPredicates(result)
    }
  }

  /** Struct-typed output of an explode/posexplode over an array of structs. */
  private def structOutput(g: Generate): Option[(Attribute, ArrayType, StructType)] = {
    val (child, outIdx) = g.generator match {
      case e: Explode => (e.child, 0)
      case p: PosExplode => (p.child, 1)
      case _ => return None
    }
    child.dataType match {
      case at @ ArrayType(st: StructType, _) if st.nonEmpty &&
          g.generatorOutput.length > outIdx =>
        Some((g.generatorOutput(outIdx), at, st))
      case _ => None
    }
  }

  private def candidates(plan: LogicalPlan): Seq[ExprId] = {
    val res = mutable.ArrayBuffer[ExprId]()
    plan.foreachWithSubqueries {
      case g: Generate => structOutput(g).foreach { case (out, _, _) => res += out.exprId }
      case _ =>
    }
    res.toSeq
  }

  /**
   * Field ordinals of `id` used anywhere in the plan, or None when `id` is used in any other
   * way than `id.field` or a plain forward through a Project list.
   */
  private def usedOrdinals(plan: LogicalPlan, id: ExprId): Option[Set[Int]] = {
    val used = mutable.Set[Int]()
    var ok = true

    def walk(e: Expression): Unit = e match {
      case GetStructField(a: Attribute, ord, _) if a.exprId == id => used += ord
      case a: Attribute if a.exprId == id => ok = false
      case other => other.children.foreach(walk)
    }

    def checkOutput(p: LogicalPlan): Unit = {
      if (p.output.exists(_.exprId == id)) ok = false
    }

    checkOutput(plan)
    plan.foreachWithSubqueries { node =>
      // The struct may only flow through operators that either do not look at it or only
      // look at it through the expressions checked below. Anything else (writes, set
      // operations, DISTINCT, CTE definitions, ...) could observe the whole struct.
      if (!transparent(node) && node.children.exists(_.output.exists(_.exprId == id))) {
        ok = false
      }
      node match {
        case g: Generate if g.generatorOutput.exists(_.exprId == id) =>
          walk(g.generator)
        case p: Project =>
          p.projectList.foreach {
            case a: Attribute if a.exprId == id => // forwarded unchanged
            case e => walk(e)
          }
        case other =>
          other.expressions.foreach(walk)
      }
      // The output of an EXISTS subquery is never observed, all other subquery outputs are.
      node.expressions.foreach(_.foreach {
        case _: Exists =>
        case s: SubqueryExpression => checkOutput(s.plan)
        case _ =>
      })
    }
    if (ok) Some(used.toSet) else None
  }

  private def transparent(p: LogicalPlan): Boolean = p match {
    case _: Project | _: Filter | _: Join | _: Generate | _: SubqueryAlias | _: Sort |
         _: LocalLimit | _: GlobalLimit | _: Aggregate | _: Window | _: Repartition |
         _: RepartitionByExpression | _: Sample | _: ResolvedHint => true
    case _ => false
  }

  private def pruneOne(plan: LogicalPlan, id: ExprId): LogicalPlan = {
    var target: Option[(Generate, Attribute, ArrayType, StructType)] = None
    plan.foreachWithSubqueries {
      case g: Generate => structOutput(g).foreach { case (out, at, st) =>
        if (out.exprId == id) target = Some((g, out, at, st))
      }
      case _ =>
    }
    val (gen, out, at, st) = target.getOrElse(return plan)
    val used = usedOrdinals(plan, id).getOrElse(return plan).toSeq.sorted
    if (used.isEmpty || used.length == st.length) return plan

    val arr = gen.generator.children.head
    val fields = used.map { o =>
      GetArrayStructFields(arr, st(o), o, st.length, at.containsNull || st(o).nullable)
    }
    // One field: explode(arr.f) and use the generated value directly, as Spark does when the
    // generate sits right below a Project. Several fields: explode(arrays_zip(arr.f1, ...)).
    val (newChild, newOut, fieldRef) = if (fields.length == 1) {
      val newOut = AttributeReference(out.name, st(used.head).dataType, nullable = true,
        out.metadata)(out.exprId, out.qualifier)
      (fields.head, newOut, (a: Attribute, _: Int) => newOut.withQualifier(a.qualifier))
    } else {
      val zipped = new ArraysZip(fields)
      val newSt = zipped.dataType.asInstanceOf[ArrayType].elementType.asInstanceOf[StructType]
      val newOrd = used.zipWithIndex.toMap
      val newOut = AttributeReference(out.name, newSt, out.nullable, out.metadata)(
        out.exprId, out.qualifier)
      (zipped, newOut, (a: Attribute, ord: Int) => {
        val i = newOrd(ord)
        GetStructField(newOut.withQualifier(a.qualifier), i, Some(newSt(i).name))
      })
    }
    val newGenerator = gen.generator match {
      case _: Explode => Explode(newChild)
      case _: PosExplode => PosExplode(newChild)
    }

    plan.transformUpWithSubqueries {
      case g: Generate if g.generatorOutput.exists(_.exprId == id) =>
        g.copy(generator = newGenerator,
          generatorOutput = g.generatorOutput.map(a => if (a.exprId == id) newOut else a))
      case node => node.transformExpressionsDown {
        case GetStructField(a: Attribute, ord, _) if a.exprId == id => fieldRef(a, ord)
        case a: Attribute if a.exprId == id => newOut.withQualifier(a.qualifier)
      }
    }
  }

  /** The array all fields of a zip were projected from, if they all come from one. */
  private def zipBase(z: ArraysZip): Option[Expression] = z.children match {
    case Seq(first: GetArrayStructFields, rest @ _*) if rest.forall {
      case f: GetArrayStructFields => f.child.semanticEquals(first.child)
      case _ => false
    } => Some(first)
    case _ => None
  }

  private def simplifyZipPredicates(plan: LogicalPlan): LogicalPlan = {
    plan.transformAllExpressionsWithSubqueries {
      case s @ Size(z: ArraysZip, _) if zipBase(z).isDefined => s.copy(child = zipBase(z).get)
      case IsNotNull(z: ArraysZip) if zipBase(z).isDefined => IsNotNull(zipBase(z).get)
      case IsNull(z: ArraysZip) if zipBase(z).isDefined => IsNull(zipBase(z).get)
    }
  }
}
