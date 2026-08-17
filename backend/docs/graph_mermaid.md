---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	route(route)
	supervisor(supervisor)
	task_dispatcher(task_dispatcher)
	dependency_blocker(dependency_blocker)
	rag_agent(rag_agent)
	market_agent(market_agent)
	arbiter_agent(arbiter_agent)
	batch_reflection(batch_reflection)
	synthesizer(synthesizer)
	compliance(compliance)
	commit_answer(commit_answer)
	compliance_failure_handler(compliance_failure_handler)
	planning_failure_handler(planning_failure_handler)
	sensitive_refusal(sensitive_refusal)
	out_of_scope_refusal(out_of_scope_refusal)
	direct_answer(direct_answer)
	agent_error_handler(agent_error_handler)
	__end__([<p>__end__</p>]):::last
	__start__ --> route;
	agent_error_handler --> batch_reflection;
	arbiter_agent --> batch_reflection;
	batch_reflection -.-> agent_error_handler;
	batch_reflection -.-> arbiter_agent;
	batch_reflection -.-> dependency_blocker;
	batch_reflection -.-> market_agent;
	batch_reflection -.-> rag_agent;
	batch_reflection -.-> synthesizer;
	batch_reflection -.-> task_dispatcher;
	compliance -. &nbsp;end&nbsp; .-> commit_answer;
	compliance -. &nbsp;compliance_failure&nbsp; .-> compliance_failure_handler;
	compliance -. &nbsp;synthesizer_retry&nbsp; .-> synthesizer;
	dependency_blocker -.-> synthesizer;
	dependency_blocker -.-> task_dispatcher;
	direct_answer --> compliance;
	market_agent --> batch_reflection;
	rag_agent --> batch_reflection;
	route -.-> direct_answer;
	route -.-> out_of_scope_refusal;
	route -.-> sensitive_refusal;
	route -.-> supervisor;
	supervisor -.-> agent_error_handler;
	supervisor -.-> arbiter_agent;
	supervisor -.-> dependency_blocker;
	supervisor -.-> market_agent;
	supervisor -. &nbsp;planning_failure&nbsp; .-> planning_failure_handler;
	supervisor -.-> rag_agent;
	supervisor -.-> synthesizer;
	supervisor -.-> task_dispatcher;
	synthesizer --> compliance;
	task_dispatcher -.-> agent_error_handler;
	task_dispatcher -.-> arbiter_agent;
	task_dispatcher -.-> market_agent;
	task_dispatcher -.-> rag_agent;
	task_dispatcher -.-> synthesizer;
	commit_answer --> __end__;
	compliance_failure_handler --> __end__;
	out_of_scope_refusal --> __end__;
	planning_failure_handler --> __end__;
	sensitive_refusal --> __end__;
	dependency_blocker -.-> dependency_blocker;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
