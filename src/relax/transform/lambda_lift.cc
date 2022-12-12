/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module) : ExprMutator(module) { mod_ = module; }

  using ExprMutator::VisitExpr_;

  void VisitBinding_(const VarBindingNode* binding) final {
    bool is_lambda = false;
    if (auto func = binding->value.as<FunctionNode>()) {
      if (!func->HasNonzeroAttr(attr::kPrimitive)) {
        is_lambda = true;
        recur_vars_.push_back(binding->var);
      }
    }

    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);

    LOG(INFO) << "\nyongwww emit new_var: " << new_var->name_hint() << " VAR: " << new_var
              << " \n vid: " << new_var->vid << " checked_type: " << new_var->checked_type_
              << " \n emit new_var shape:  " << new_var->shape_ << "\n new_value: " << new_value
              << " new_value shape: " << new_value->shape_;

    auto emit = [this](VarBinding b) {
      Var test;
      if (this->builder_->CurrentBlockIsDataFlow() && !b->var.as<DataflowVarNode>()) {
        LOG(INFO) << "get here 68";
        test = this->builder_->EmitOutput(b);
      } else {
        LOG(INFO) << "get here 72";
        test = this->builder_->Emit(b);
      }
      LOG(INFO) << "\nyongwww emit varbinding:  " << test << " name: " << test->name_hint()
                << "\nvid: " << test->vid << " checked_type: " << test->checked_type_
                << "\n shape: " << test->shape_;
    };

    if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      emit(GetRef<VarBinding>(binding));
      return;
    }

    LOG(INFO) << "\nyongwww emit old_var: " << binding->var
              << " name: " << binding->var->name_hint() << "\nvid: " << binding->var->vid
              << " checked_type: " << binding->var->checked_type_
              << "\n shape: " << binding->var->shape_;
    if (!recur_vars_.empty() && recur_vars_.back() == new_var) {
      new_var->checked_type_ = new_value->checked_type_;
      new_var->shape_ = new_value->shape_;
      LOG(INFO) << "DEBUG 83, var: " << new_var << "  vid: " << new_var->vid
                << "  name:" << new_var->name_hint() << " shape: " << new_var->shape_;
    }

    LOG(INFO) << "83  yongwww emit new_var: " << new_var << " name:" << new_var->name_hint()
              << " vid: " << new_var->vid << " checked_type: " << new_var->checked_type_
              << " shape: " << new_var->shape_;

    if (new_value->shape_ == nullptr || new_value->shape_ == (nullptr)) {
      LOG(INFO) << "yongwww new_value get here";
      // todo (yongwww): working
      // new_value->shape_ = RuntimeDepShape();
    }

    emit(VarBinding(new_var, new_value));

    if (is_lambda) {
      recur_vars_.pop_back();
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto const* var_node = call_node->op.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      bool has_closure = HasClosure(var);
      auto val = builder_->LookupBinding(var);
      LOG(INFO) << "106 yongwww var " << var->name_hint() << " \n--- var: " << var
                << " \n--- vid: " << var->vid << " \ntype: " << var->checked_type_
                << "  \nhas_closure: " << has_closure << " \n val: " << val;
      // Call "relax.invoke_closure" to invoke closure
      if (has_closure && val->IsInstance<CallNode>()) {
        Var clo_arg = var;
        if (this->var_remap_.find(var->vid) != this->var_remap_.end()) {
          clo_arg = this->var_remap_.at(var->vid);
        }
        return Call(invoke_closure_op_, {clo_arg, Tuple(call_node->args)}, {},
                    {GetStructInfo(GetRef<Expr>(call_node))});
      }
      auto it = lambda_map_.find(var);
      if (it != lambda_map_.end()) {
        LOG(INFO) << "get here it->second: " << it->second;
        // flatten nested call, e.g. call(y)(x) -> call(x, y))
        Array<relay::Expr> new_args;
        for (const auto arg : call->args) {
          new_args.push_back(arg);
        }
        if (const auto* nest_call = it->second.as<CallNode>()) {
          for (const auto arg : nest_call->args) {
            new_args.push_back(arg);
          }
          return Call(nest_call->op, new_args, call_node->attrs, call_node->sinfo_args);
        }
        LOG(INFO) << "130 get here it->second: " << it->second;
        return Call(it->second, call->args, call_node->attrs, call_node->sinfo_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // TODO(@yongwww): consider appending inner func name into the lifted func name
    String lift_func_name = "lifted_func_" + std::to_string(lift_func_num_++);
    auto global = GlobalVar(lift_func_name);
    Array<Var> free_vars = FreeVars(func);
    Array<Var> captured_vars;

    Array<Var> typed_captured_vars;
    bool recursive = false;
    // recur_vars_.push_back(free_vars[1])
    for (const auto& var : free_vars) {
      if (!recur_vars_.empty() && var == recur_vars_.back()) {
        recursive = true;
      } else {
        captured_vars.push_back(var);
      }
    }

    Map<Var, Expr> rebinding_map;
    for (auto free_var : captured_vars) {
      Var var = Var(free_var->name_hint(), GetStructInfo(free_var), free_var->span);
      // var->shape_ = free_var->shape_; todo (yongwww)
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }
    // recursive call
    LOG(INFO) << "recur_vars_ size: " << recur_vars_.size() << "  recursive: " << recursive;
    if (recursive) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        // checked_type_ is required by block_blocker, it will be reset later
        UpdateType(global, recur_vars_.back()->checked_type());
        LOG(INFO) << "DEBUG 181 yongwww recur_vars_.back()->shape_: " << recur_vars_.back()->shape_
                  << " recur_vars_.back()->checked_type_: " << recur_vars_.back()->checked_type_;
        // UpdateType(global, recur_vars_.back()->checked_type());
        // UpdateType(global, free_vars[0]->checked_type_); // todo(yongwww): Update here
        lambda_map_.emplace(recur_vars_.back(), Call(global, fvs));
      } else {
        LOG(INFO) << "nothing here right?";
        if (recur_vars_.size() > 0) {
          lambda_map_.emplace(recur_vars_.back(), global);
        }
      }
    }

    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : func_node->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }
    Expr body = this->VisitWithNewScope(func_node->body);
    Expr visited_func;

    if (all_params_unchanged && body.same_as(func_node->body)) {
      visited_func = GetRef<Expr>(func_node);
    } else if (const auto& body_sinfo = MatchStructInfo<ObjectStructInfo>(body)) {
      visited_func = Function(params, body, body_sinfo.value(), func_node->attrs);
      // Function(params, body, body->checked_type_, RuntimeDepShape(), func_node->attrs)
    } else {
      visited_func = Function(params, body, func_node->ret_struct_info, func_node->attrs);
      // Function(params, body, func_node->ret_type, func_node->ret_shape, func_node->attrs);
    }
    auto new_func = Downcast<Function>(visited_func);

    Function lifted_func;
    bool is_closure = IsClosure(captured_vars);
    if (!is_closure) {
      lifted_func = Function(
          /*params=*/new_func->params,
          /*body=*/new_func->body,
          /*ret_struct_info=*/new_func->ret_struct_info,
          /*attrs=*/new_func->attrs,
          /*span=*/new_func->span);
    } else {
      // Flatten the Closure
      std::vector<Var> closure_params;
      closure_params.reserve(func->params.size() + typed_captured_vars.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        closure_params.emplace_back(func->params[i]);
      }
      for (size_t i = 0; i < typed_captured_vars.size(); ++i) {
        closure_params.emplace_back(typed_captured_vars[i]);
      }

      lifted_func = Function(/*params=*/closure_params,
                             /*body=*/Bind(new_func->body, rebinding_map),
                             /*ret_struct_info=*/new_func->ret_struct_info,
                             /*attrs=*/new_func->attrs,
                             /*span=*/func->span);

      for (Var param : closure_params) {
        CHECK(param->checked_type_.defined())
            << "relax.Function requires params to contain checked_type_";
      }
    }

    ICHECK(lifted_func.defined());

    // Add the lifted function to the module.
    UpdateStructInfo(global, GetStructInfo(lifted_func));
    // global->checked_type_ = lifted_func->checked_type_;
    // global->shape_ = lifted_func->shape_;
    builder_->UpdateFunction(global, lifted_func);
    LOG(INFO) << "is_closure: " << is_closure;

    if (!is_closure) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      // Call make_closure intrinsic
      return Call(make_closure_op_, {global, Tuple(fvs)}, {}, {});
    }
  }

  bool HasClosure(const Var& var) {
    auto val = builder_->LookupBinding(var);
    if (const auto* value = val.as<GlobalVarNode>()) {
      IRModule ctx_mod = builder_->GetContextIRModule();
      ICHECK(ctx_mod->functions.size() > 0);
      BaseFunc func = ctx_mod->Lookup(GetRef<GlobalVar>(value));
      if (const auto* func_node = func.as<FunctionNode>()) {
        if (const auto* call_node = func_node->body.as<CallNode>()) {
          if (call_node->op == make_closure_op_) {
            return true;
          }
        } else if (const auto* seq_expr_node = func_node->body.as<SeqExprNode>()) {
          // the return var points to a make_closure intrinsic
          if (const auto* var = seq_expr_node->body.as<VarNode>()) {
            return HasClosure(GetRef<Var>(var));
          }
        }
      }
    } else if (const auto* func_node = val.as<FunctionNode>()) {
      if (const auto* call_node = func_node->body.as<CallNode>()) {
        if (call_node->op == make_closure_op_) {
          return true;
        }
      }
    } else if (const auto* call_node = val.as<relax::CallNode>()) {
      // recursive call
      auto op = call_node->op;
      if (make_closure_op_ == op) {
        return true;
      }
      if (const auto* lv = op.as<VarNode>()) {
        return HasClosure(GetRef<Var>(lv));
      }
    }
    return false;
  }

  bool IsClosure(const Array<Var>& captured_vars) { return captured_vars.size() > 0; }

  IRModule Lift() {
    auto glob_funcs = mod_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_struct_info, func->attrs);
        builder_->UpdateFunction(pair.first, func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  Array<Var> recur_vars_;
  IRModule mod_;
  Type rec_type;
  size_t lift_func_num_ = 0;
  /*! \brief Cache ops that would be used later to reduce lookup overhead. */
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::LambdaLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
