主要问题

  1. 严重：并发分支会互相覆盖 plan 状态，导致任务结果丢失。
     dispatch_tasks 给每个 Send 分支传入整份 plan 快照；每个 Agent 完成后又返回整份 plan。merge_plan 按 task_id 以右侧结
     果覆盖，因此最后合并的分支会把其他分支已完成任务覆盖回 running。已用当前 reducer 复现：t1=completed、t2=completed
     最终会变成 t1=running、t2=completed。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/multi_agent_controller.py:104、/E:/pythonprojects/
     fund_analyser/backend/app/agent/retrieval_agent.py:437、/E:/pythonprojects/fund_analyser/backend/app/agent/
     state_reducers.py:59。
     建议：并发 Agent 只能返回自身任务的 TaskPatch，不要返回全量 plan；reducer 应按字段合并任务更新，并禁止终态被
     running/pending 覆盖。调度器的 running 状态也应先写入主状态，而不是仅存在于分支副本中。

  2. 严重：Agent 没有可用工具时，任务不会进入终态。
     retrieval_agent 在无工具时只写 sub_results，没有把任务标为 failed，也没有加入 failed_tasks。该任务会残留为 running/
     pending，随后调度器又因没有 ready task 直接进入 synthesizer，最终答案会把失败伪装成“未执行”。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/retrieval_agent.py:332。
     建议：所有提前返回和异常路径统一走 _finish_task，保证写入终态、错误码、耗时和结果。

  3. 高：依赖失败仍会调度下游任务，且没有失败/跳过语义。
     get_ready_tasks 把 completed | failed 都视为依赖满足。若下游必须依赖上游数据，它会在输入缺失时继续执行。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/multi_agent_state.py:174。
     建议：增加 skipped/blocked 状态；默认仅成功依赖可执行。只有明确声明“可容忍失败”的任务才允许继续。

  4. 高：Supervisor 的计划缺少硬校验，非法 DAG 会静默提前汇总。
     当前只做了有限字段默认值填充，没有校验任务 ID 唯一性、依赖是否存在、循环依赖、Agent 白名单、task_type 与 Agent 的匹
     配、基金代码格式等。缺失依赖或环会让所有任务都不 ready，随后进入 synthesizer。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/supervisor.py:217、/E:/pythonprojects/fund_analyser/
     backend/app/agent/multi_agent_controller.py:158。
     另外，prompt 示例里出现了两个同名 "plan" 键，JSON 实际只保留后一个，容易诱导模型输出错误结构。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/supervisor.py:61。
     建议：用 Pydantic schema 验证并规范化计划；校验失败时重试规划或生成明确失败答案，不进入执行图。

  5. 高：路由结果除 sensitive 外没有真正参与图分支。
     chitchat、out_of_scope、general_finance 都会进入 Supervisor，再可能进入 RAG/市场工具。这既增加成本，也可能让越界问
     题得到不该有的基金检索回答。
     位置：/E:/pythonprojects/fund_analyser/backend/app/services/router.py:84、/E:/pythonprojects/fund_analyser/backend/
     app/agent/multi_agent_controller.py:72。
     建议：明确路由为 direct_answer、out_of_scope_refusal、fund_workflow、sensitive_refusal。只有需要外部数据的 intent
     才进入规划图。

  6. 中：全局反思只比较“本批新完成任务”，跨批次冲突会漏检。
     上游任务在第一批完成并被标记 reflected=True 后，第二批下游任务不会再与它比较；有依赖的任务链尤其容易漏掉冲突。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/reflection_agent.py:94、/E:/pythonprojects/fund_analyser/
     backend/app/agent/reflection_agent.py:137。
     建议：按领域实体/基金代码/指标维护“已验证事实”，让新结果与相关历史事实比对；或只在最终汇总前做一次完整一致性检查。

  7. 中：合规重试预算跨对话轮次泄漏。
     新计划重置了大部分执行字段，却没有重置 compliance_retry_count。上一轮曾触发合规重试后，下一轮第一次违规可能直接走兜
     底拒绝。
     位置：/E:/pythonprojects/fund_analyser/backend/app/agent/supervisor.py:268、/E:/pythonprojects/fund_analyser/
     backend/app/agent/multi_agent_controller.py:187。
     建议：新 round 初始化时显式重置该字段；将“草稿合成”和“通过合规后归档”拆开，避免每次重试都写入 plan_history。

  8. 中：SSE 的并发任务观测不正确。
     agent_end 从全量 plan 猜测任务，并最终固定发送 "task_id": ""。多个同类 Agent 并发时，前端无法可靠关联开始、结束和工
     具调用。
     位置：/E:/pythonprojects/fund_analyser/backend/app/tasks/chat_tasks.py:213。
     建议：在 Send 时生成 task_id 关联上下文，或从 LangGraph run metadata 透传；agent_end 直接使用当前分支 task ID。