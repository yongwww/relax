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
    LOG(INFO) << " \n58 yongwww binding->var: " << binding->var->name_hint()
              << " type: " << binding->var->checked_type_
              << "\n sinfo: " << binding->var->struct_info_
              <<" \nbinding->value: " << binding->value
              <<" \nnew_value: " << new_value
              << "\n GetStructInfo(new_value): " << GetStructInfo(new_value)
              << "\n new_value sinfo: " << new_value->struct_info_
              << "\n new_value.same_as(binding->value: " << new_value.same_as(binding->value);
    if (!binding->var->struct_info_.defined()) {
       UpdateStructInfo(binding->var, GetStructInfo(new_value));
      //binding->var->struct_info_ = GetStructInfo(new_value);
     //binding->var->checked_type_ = GetStaticType(GetStructInfo(new_value));
    }
    /*
    if (binding->var->name_hint() == "in_call") {
          binding->var->struct_info_ = ObjectStructInfo();
          builder_->EmitNormalized(VarBinding(binding->var, new_value));
          return;
    }
    */
    if (new_value.same_as(binding->value)) {
      LOG(INFO) << "yongwww 72";
      if (new_value->struct_info_.defined()) {
        binding->var->struct_info_ = GetStructInfo(new_value);
        binding->var->checked_type_ = new_value->checked_type_;
      }
      builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      LOG(INFO) << "yongwww 79";
      if (new_value->struct_info_.defined()) {
        LOG(INFO) << "yongwww 81";
        binding->var->struct_info_ = GetStructInfo(new_value);
        binding->var->checked_type_ = new_value->checked_type_;
        
      }
      builder_->EmitNormalized(VarBinding(binding->var, new_value));
    }
    if (is_lambda) {
      recur_vars_.pop_back();
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    LOG(INFO) << "\n call_node op is "
              << call_node->op << " sinfo: " << call_node->struct_info_;
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    LOG(INFO) << "\n 94 call: "<< call << " \n ###   call sinfo: " << call->struct_info_;
    if (auto const* var_node = call_node->op.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      bool has_closure = HasClosure(var);
      auto val = builder_->LookupBinding(var);
      LOG(INFO) << "106 yongwww var " << var->name_hint() << " \n--- var: " << var
                << " \n--- vid: " << var->vid << " \ntype: " << var->struct_info_
                << "  \nhas_closure: " << has_closure << " \n val: " << val;

      if (var->name_hint() == "outer_func") { // hack
        call->struct_info_ = ObjectStructInfo();
        call->checked_type_ = ObjectType();
      }
      if (this->var_remap_.find(var->vid) != this->var_remap_.end()) { // nothing happens
          Var test = this->var_remap_.at(var->vid);
          LOG(INFO) << "test var: " << test << " name: " << test->name_hint();
      }
      // Call "relax.invoke_closure" to invoke closure
      if (has_closure && val->IsInstance<CallNode>()) { // nothing related
        Var clo_arg = var;
        if (this->var_remap_.find(var->vid) != this->var_remap_.end()) {
          clo_arg = this->var_remap_.at(var->vid);
        }
        return Call(invoke_closure_op_, {clo_arg, Tuple(call_node->args)}, {},
                    {GetStructInfo(GetRef<Expr>(call_node))});
      }
      //if (val->IsInstance<GlobalVarNode>()) {
      //}
      auto it = lambda_map_.find(var);
      if (it != lambda_map_.end()) { // didn't reach here
        LOG(INFO) << "get here it->second: " << it->second;
        // flatten nested call, e.g. call(y)(x) -> call(x, y))
        Array<relay::Expr> new_args;
        Array<StructInfo> params;
        for (const auto arg : call->args) {
          new_args.push_back(arg);
          params.push_back(StructInfoFromType(arg->checked_type()));
        }
        if (const auto* nest_call = it->second.as<CallNode>()) {
          // Update the StructInfo accordingly
          // FuncStructInfo
          
          for (const auto arg : nest_call->args) {
            new_args.push_back(arg);
            params.push_back(StructInfoFromType(arg->checked_type()));
          }
          LOG(INFO) << " nest_call->op: " << nest_call->op
                    << "\n nest_call->op sinfo: " << nest_call->op->struct_info_
                    << "\n call-> sinfo: " << nest_call->struct_info_;
          // StructInfo ret = StructInfoFromType(func_type->ret_type);
          StructInfo new_func_sinfo;
          if (const auto* fsinfo = nest_call->op->struct_info_.as<FuncStructInfoNode>()) {
            auto func_sinfo = GetRef<FuncStructInfo>(fsinfo);
            new_func_sinfo = FuncStructInfo(params, func_sinfo->ret);
          }
          // UpdateStructInfo(nest_call->op, new_func_sinfo);
          nest_call->op->struct_info_ = new_func_sinfo;
          LOG(INFO) << "\n updated nest_call->op sinfo: " << nest_call->op->struct_info_
                    << "\n call-> sinfo: " << nest_call->struct_info_;
          return Call(nest_call->op, new_args, call_node->attrs, call_node->sinfo_args);
        }
        LOG(INFO) << "130 get here it->second: " << it->second;
        return Call(it->second, call->args, call_node->attrs, call_node->sinfo_args);
      }
      // builder_->Emit(call_node);
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
        // UpdateType(global, recur_vars_.back()->checked_type());
        // global->checked_type_ = recur_vars_.back()->checked_type();
        UpdateStructInfo(global, GetStructInfo(recur_vars_.back()));
        // UpdateStructInfo(global, GetStructInfo(lifted_func));
        LOG(INFO) << "DEBUG 181 yongwww recur_vars_.back()->shape_: " //<< recur_vars_.back()->shape_
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
    global->struct_info_ = GetStructInfo(lifted_func);
    // UpdateStructInfo(global, GetStructInfo(lifted_func));
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
  Array<Var> recur_vars_; // todo: -> lifted_vars
  IRModule mod_;
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
