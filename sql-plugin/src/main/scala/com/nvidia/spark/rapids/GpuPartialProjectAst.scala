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

import ai.rapids.cudf.{ast, ColumnVector, Scalar, Table}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.shims.ShimUnaryExpression

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuPartialProjectAst {
  private val eligible = TreeNodeTag[Boolean]("rapids.project.partialAst.eligible")

  /** Only called for whole outputs approved by ExprMeta.canThisBeAst. */
  def markEligible(expr: Expression): Unit = {
    // Remember eligible shared subexpressions as well: CSE may move them to earlier tiers.
    // TreeNode.mapChildren preserves tags when replacing children with common references.
    expr.foreach(_.setTagValue(eligible, true))
  }

  private def wrap(expr: GpuExpression): GpuExpression = expr match {
    case alias: GpuAlias =>
      // CSE creates new aliases without the original expression's tags.
      alias.withNewChildren(Seq(wrap(alias.child.asInstanceOf[GpuExpression])))
        .asInstanceOf[GpuExpression]
    case _: GpuBoundReference | _: GpuLiteral => expr
    case _ if expr.getTagValue(eligible).contains(true) &&
        GpuBatchUtils.isFixedWidth(expr.dataType) =>
      // Combined-expression replacement may have introduced a non-AST implementation.
      // Validate conversion after all substitutions, without compiling native resources.
      val convertible = try {
        expr.convertToAst(Int.MaxValue)
        true
      } catch {
        case _: IllegalStateException => false
      }
      if (convertible) GpuPartialProjectAst(expr) else expr
    case _ => expr
  }

  def wrapTieredProject(project: GpuTieredProject, conf: SQLConf): GpuTieredProject = {
    if (RapidsConf.ENABLE_PROJECT_PARTIAL_AST.get(conf)) {
      GpuTieredProject(project.exprTiers.map(_.map(wrap).toList))
    } else {
      project
    }
  }
}

/**
 * Evaluate one bound, metadata-approved expression with AST inside the ordinary tiered project.
 * Inserted after CSE and binding, so ordinary retry and nondeterministic checkpoints still apply
 * to the complete projection. Direct references and literals retain their existing evaluation.
 */
case class GpuPartialProjectAst(child: Expression) extends ShimUnaryExpression with GpuExpression {
  override def dataType: DataType = child.dataType
  override def nullable: Boolean = child.nullable

  // Native state must never be serialized from the driver. A task owns the compiled expression,
  // including when a downstream limit stops consuming the projection early.
  @transient private var compiled: ast.CompiledExpression = _

  private def compile(): ast.CompiledExpression =
    child.asInstanceOf[GpuExpression].convertToAst(Int.MaxValue).compile()

  private def evaluate(batch: ColumnarBatch,
      expression: ast.CompiledExpression): GpuColumnVector = {
    def compute(table: Table): GpuColumnVector = {
      withResource(expression.computeColumn(table)) { result =>
        GpuColumnVector.from(result.incRefCount(), dataType)
      }
    }
    if (batch.numCols() > 0) {
      withResource(GpuColumnVector.from(batch))(compute)
    } else {
      // cuDF needs a row-count carrier for expressions evaluated on count-only batches.
      withResource(Scalar.fromBool(false)) { scalar =>
        withResource(ColumnVector.fromScalar(scalar, batch.numRows())) { column =>
          withResource(new Table(column))(compute)
        }
      }
    }
  }

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    Option(TaskContext.get()) match {
      case Some(context) =>
        if (compiled == null) {
          compiled = closeOnExcept(compile()) { expression =>
            onTaskCompletion(context) {
              withResource(expression) { _ => compiled = null }
            }
            expression
          }
        }
        evaluate(batch, compiled)
      case None =>
        // Direct callers (including unit tests) do not have a task resource owner.
        withResource(compile())(evaluate(batch, _))
    }
  }
}
