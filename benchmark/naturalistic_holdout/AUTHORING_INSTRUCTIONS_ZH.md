# Naturalistic Query Holdout 填写说明

这份说明写给填写 `authoring_tasks_seed22_n80.csv` 的人：需要补齐的是
`naturalistic_query` 一列，也就是把每个任务改写成工程师真正会问出口的话。

## 基本规则

- 用真实工程提问的语气写，不要照抄模板化句式。
- 保持任务含义不变：网络、操作、分析类型、输出量都要和原任务一致。
- query 里不出现答案、ground truth 或 reference code 的信息。
- 可以使用自然的省略、上下文语气和工程表达，例如“帮我看一下”“这个工况下”
  “最后给我报一下”。
- 自然不等于歧义：写完之后这条 query 仍然只能对应一种执行方式。
- 每条 query 写完后在 `author` 填上名字或缩写，`status` 先保留 `draft`。

## 审阅规则

第二个人审阅时确认三件事：

- query 的任务含义与原任务一致；
- query 的表达是重写的，而不是 `task_brief` 的固定说法；
- query 中没有答案信息。

三点都通过后，把 `status` 改为 `accepted`、`reviewed` 或 `final`。只有这三
种状态的行会进入最终的 holdout JSON。

## 命名

holdout 的叫法取决于谁写、谁审。由作者填写、作者组审阅的，称为：

> author-curated naturalistic-query holdout

电力系统／能源系统领域专家实际参与撰写或审阅的，才称为：

> expert-reviewed naturalistic-query holdout

另外，holdout 回答的问题和“benchmark 已人工验证”不是同一个问题，两者不能
互相替代。
