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

import scala.util.Try

import com.nvidia.spark.rapids.shims.LogicalPlanShims

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Average, Complete, Count, Max, Min, Sum}
import org.apache.spark.sql.catalyst.plans.{Cross, Inner}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types._

/**
 * A Cartesian product of global aggregates always contains exactly one row, even for empty input.
 * If the inputs differ only by safe filters/projections over the same relation, compute their
 * aggregates in one scan using aggregate FILTER clauses. Run before RewriteDistinctAggregates so
 * Spark can retain the semantics of different filters on the same DISTINCT argument.
 */
private[rapids] object GpuMergeGlobalAggregates extends Rule[LogicalPlan] {
  private case class Input(base: LogicalPlan, attributes: Map[ExprId, Attribute],
      predicates: Seq[Expression])

  private case class Branch(aggregate: Aggregate, input: Input)

  private def remap(expression: Expression, attributes: Map[ExprId, Attribute]): Expression = {
    expression.transform {
      case attribute: Attribute => attributes.getOrElse(attribute.exprId, attribute)
    }
  }

  // Only fold literal arithmetic/casts. Do not evaluate arbitrary foldable expressions such as
  // query-time values, and leave throwing constants alone so an otherwise empty input stays safe.
  private def foldConstants(expression: Expression): Expression = expression.transformUp {
    case value if value.children.nonEmpty && value.children.forall(_.isInstanceOf[Literal]) &&
        value.dataType.isInstanceOf[NumericType] &&
        (value.isInstanceOf[Add] || value.isInstanceOf[Subtract] ||
          value.isInstanceOf[Multiply] || value.isInstanceOf[Cast]) =>
      Try(Literal.create(value.eval(), value.dataType)).getOrElse(value)
  }

  private def prepare(expression: Expression, attributes: Map[ExprId, Attribute]): Expression = {
    // Expand the analyzed replacement, which already contains Spark's type coercions. Inputs
    // duplicated by inlining must still pass the deterministic/nonthrowing checks below.
    foldConstants(remap(LogicalPlanShims.inlinePredicateExpressions(expression), attributes))
  }

  private def integralDigits(dataType: DataType): Option[Int] = dataType match {
    case ByteType => Some(3)
    case ShortType => Some(5)
    case IntegerType => Some(10)
    case LongType => Some(19)
    case _ => None
  }

  private def safeCast(from: DataType, to: DataType): Boolean = (from, to) match {
    case (left, right) if left == right => true
    case (left: DecimalType, right: DecimalType) =>
      right.scale >= left.scale && right.precision - right.scale >= left.precision - left.scale
    case (left, right: DecimalType) =>
      right.scale >= 0 && integralDigits(left).exists(_ <= right.precision - right.scale)
    case (FloatType, DoubleType) => true
    case _ =>
      integralDigits(from).exists(left => integralDigits(to).exists(_ >= left))
  }

  private def safeValue(expression: Expression): Boolean = expression match {
    case _: Attribute | _: Literal => true
    case cast: Cast => safeValue(cast.child) && safeCast(cast.child.dataType, cast.dataType)
    case _ => false
  }

  private def safePredicate(expression: Expression): Boolean = expression match {
    case _: Literal => expression.dataType == BooleanType
    case attribute: Attribute => attribute.dataType == BooleanType
    case And(left, right) => safePredicate(left) && safePredicate(right)
    case Or(left, right) => safePredicate(left) && safePredicate(right)
    case Not(child) => safePredicate(child)
    case comparison: BinaryComparison => comparison.children.forall(safeValue)
    case IsNull(child) => safeValue(child)
    case IsNotNull(child) => safeValue(child)
    case in: In => safeValue(in.value) && in.list.forall(safeValue)
    case _ => false
  }

  private def conjuncts(expression: Expression): Seq[Expression] = expression match {
    case And(left, right) => conjuncts(left) ++ conjuncts(right)
    case other => Seq(other)
  }

  private def normalize(plan: LogicalPlan): Option[Input] = plan match {
    case alias: SubqueryAlias => normalize(alias.child)
    case view: View =>
      normalize(view.child).flatMap { input =>
        if (view.output.size == view.child.output.size &&
            view.output.zip(view.child.output).forall { case (left, right) =>
              left.dataType == right.dataType && input.attributes.contains(right.exprId)
            }) {
          Some(input.copy(attributes = view.output.zip(view.child.output).map {
            case (left, right) => left.exprId -> input.attributes(right.exprId)
          }.toMap))
        } else {
          None
        }
      }
    case project: Project =>
      normalize(project.child).flatMap { input =>
        val attributes = project.projectList.map {
          case attribute: Attribute =>
            input.attributes.get(attribute.exprId).map(attribute.exprId -> _)
          case alias @ Alias(attribute: Attribute, _) =>
            input.attributes.get(attribute.exprId).map(alias.exprId -> _)
          case _ => None
        }
        if (attributes.forall(_.isDefined)) {
          Some(input.copy(attributes = attributes.flatten.toMap))
        } else {
          None
        }
      }
    case filter: Filter =>
      normalize(filter.child).flatMap { input =>
        val predicate = prepare(filter.condition, input.attributes)
        if (predicate.deterministic && safePredicate(predicate) &&
            predicate.references.subsetOf(input.base.outputSet)) {
          Some(input.copy(predicates = input.predicates ++ conjuncts(predicate)))
        } else {
          None
        }
      }
    case relation: LogicalRelation if !relation.isStreaming && (relation.relation match {
      case fs: HadoopFsRelation => fs.fileFormat.isInstanceOf[ParquetFileFormat]
      case _ => false
    }) =>
      Some(Input(relation, relation.output.map(a => a.exprId -> a).toMap, Seq.empty))
    case relation: LocalRelation if !relation.isStreaming =>
      Some(Input(relation, relation.output.map(a => a.exprId -> a).toMap, Seq.empty))
    case _ => None
  }

  private def branch(plan: LogicalPlan): Option[Branch] = plan match {
    case alias: SubqueryAlias => branch(alias.child)
    case aggregate: Aggregate if aggregate.groupingExpressions.isEmpty &&
        aggregate.aggregateExpressions.nonEmpty && !aggregate.isStreaming =>
      normalize(aggregate.child).map(Branch(aggregate, _))
    case _ => None
  }

  private def leaves(plan: LogicalPlan): Seq[LogicalPlan] = plan match {
    case join: Join if join.condition.isEmpty &&
        (join.joinType == Inner || join.joinType == Cross) =>
      leaves(join.left) ++ leaves(join.right)
    case other => Seq(other)
  }

  private def supported(expression: NamedExpression): Boolean = expression match {
    case Alias(aggregate: AggregateExpression, _) =>
      val function = aggregate.aggregateFunction
      (function.isInstanceOf[Count] || function.isInstanceOf[Average] ||
        function.isInstanceOf[Sum] || function.isInstanceOf[Min] || function.isInstanceOf[Max]) &&
        function.children.forall(safeValue) && aggregate.filter.forall(safePredicate) &&
        aggregate.deterministic && aggregate.mode == Complete
    case _ => false
  }

  private def merge(join: Join): Option[LogicalPlan] = {
    val inputs = leaves(join)
    // Bound optimizer work for very large Cartesian products; a partial rewrite remains valid.
    for {
      eligible <- Some(inputs).filter(value => value.size >= 2 && value.size <= 32)
      values = eligible.flatMap(branch)
      if values.size == eligible.size
      base = values.head.input.base
      if values.forall(value => value.input.base.sameResult(base) &&
        value.input.base.output.map(a => (a.name, a.dataType, a.nullable)) ==
          base.output.map(a => (a.name, a.dataType, a.nullable)))
      normalized = values.map { value =>
        val attributes = value.input.attributes.map { case (id, attribute) =>
          val index = value.input.base.output.indexWhere(_.exprId == attribute.exprId)
          id -> base.output(index)
        }
        val expressions = value.aggregate.aggregateExpressions.map { expression =>
          prepare(expression, attributes).asInstanceOf[NamedExpression]
        }
        val baseAttributes = value.input.base.output.zip(base.output).map {
          case (left, right) => left.exprId -> right
        }.toMap
        val predicates = value.input.predicates.map(remap(_, baseAttributes))
        (expressions, predicates)
      }
      if normalized.flatMap(_._1).forall(expression => supported(expression) &&
        expression.references.subsetOf(base.outputSet))
      common = normalized.head._2.filter { predicate =>
        normalized.tail.forall(_._2.exists(_.semanticEquals(predicate)))
      }
      conditions = normalized.map { case (_, predicates) =>
        predicates.filterNot(predicate => common.exists(_.semanticEquals(predicate)))
          .reduceOption[Expression](And).getOrElse(Literal.TrueLiteral)
      }
      expressions = normalized.zip(conditions).flatMap { case ((outputs, _), predicate) =>
        outputs.map(_.transform {
          case aggregate: AggregateExpression =>
            val filter = aggregate.filter.map(And(_, predicate)).getOrElse(predicate)
            aggregate.copy(filter = Some(filter))
        }.asInstanceOf[NamedExpression])
      }
      if expressions.map(_.exprId).distinct.size == expressions.size
    } yield {
      val union = conditions.reduce[Expression](Or)
      val condition = (common :+ union).reduce[Expression](And)
      val merged = values.head.aggregate.copy(aggregateExpressions = expressions,
        child = Filter(condition, base))
      Project(join.output, merged)
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = plan.transformDown {
    case join: Join if join.resolved => merge(join).getOrElse(join)
  }
}
