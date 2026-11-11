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

import org.apache.spark.sql.catalyst.expressions.{And, ArrayExists, Attribute, Explode, Expression, IsNotNull, LambdaFunction, NamedLambdaVariable, PosExplode, PredicateHelper, SubqueryExpression}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Generate, LogicalPlan}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.types.ArrayType

/**
 * Adds an array-level copy of a filter on the output of `explode`/`posexplode` below the
 * generate:
 * {{{
 *   Filter(p(g), Generate(explode(arr), outer = false, [g]))
 *   =>
 *   Filter(p(g), Generate(explode(arr), outer = false, [g], Filter(exists(arr, x -> p(x)))))
 * }}}
 * A row whose array has no element satisfying `p` produces no output after explode and filter,
 * so dropping it before the generate does not change the result (a null array produces no rows
 * either, and `exists` is then null, which a filter drops too). The original filter stays.
 *
 * The new filter only references the generate's child, so Spark pushes it further down, below
 * joins and down to the scan. This matters when Spark has already pushed other predicates
 * (typically EXISTS / NOT EXISTS, later rewritten to semi / anti joins) below the generate: the
 * row-reducing filter on the generated value would otherwise run only after those joins, while
 * the same query on a flat schema filters before them.
 *
 * Only conjuncts that reference nothing but the generated value are copied, and only when at
 * least one of them is more than an IS NOT NULL check. Each generate is handled once (tagged).
 * Can be turned off by adding this rule's name to `spark.sql.optimizer.excludedRules`.
 */
object GeneratorArrayPrefilter extends Rule[LogicalPlan] with PredicateHelper {

  private val DONE = TreeNodeTag[Unit]("rapids.generatorArrayPrefilter")

  override def apply(plan: LogicalPlan): LogicalPlan = plan.transformUpWithSubqueries {
    case f @ Filter(cond, g: Generate) if !g.outer && g.getTagValue(DONE).isEmpty =>
      prefilter(cond, g) match {
        case Some(arrayCond) =>
          val newGen = g.copy(child = Filter(arrayCond, g.child))
          newGen.copyTagsFrom(g)
          newGen.setTagValue(DONE, ())
          f.copy(child = newGen)
        case None => f
      }
  }

  private def prefilter(cond: Expression, g: Generate): Option[Expression] = {
    val (arr, out) = g.generator match {
      case e: Explode if g.generatorOutput.length == 1 => (e.child, g.generatorOutput.head)
      case p: PosExplode if g.generatorOutput.length == 2 => (p.child, g.generatorOutput(1))
      case _ => return None
    }
    val elementType = arr.dataType match {
      case ArrayType(et, _) => et
      case _ => return None
    }
    if (!arr.deterministic) return None
    val onOutput = splitConjunctivePredicates(cond).filter { c =>
      c.deterministic && !SubqueryExpression.hasSubquery(c) && c.references.nonEmpty &&
        c.references.forall(_.exprId == out.exprId)
    }
    if (!onOutput.exists(c => !c.isInstanceOf[IsNotNull])) return None

    val x = NamedLambdaVariable("x", elementType,
      arr.dataType.asInstanceOf[ArrayType].containsNull)
    val body = onOutput.reduce(And).transform {
      case a: Attribute if a.exprId == out.exprId => x
    }
    Some(ArrayExists(arr, LambdaFunction(body, Seq(x))))
  }
}
