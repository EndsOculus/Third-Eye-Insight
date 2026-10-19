"""Fixed, private-text-free presentation of model-selected labels and evidence."""
TOPIC={'work':'工作','study':'学习','daily_life':'日常','health':'健康','other':'其他','unknown':'未知'}
TYPE={'question':'提问措辞','answer':'回答候选','acknowledgment':'确认措辞','support':'支持措辞','disagreement':'分歧措辞','joke':'玩笑候选','other':'其他','unknown':'未知'}
REASON={'direct_answer':'模型认为后文回答了前文问题','explicit_address':'模型认为存在明确指向','shared_topic':'仅有话题关联，非回应事实',
        'acknowledges_prior':'模型认为后文确认了前文','supportive_wording':'模型识别到支持性措辞','contrasting_statement':'模型认为存在相反表述',
        'insufficient_context':'上下文不足','ambiguous_targets':'存在多个可能对象','possible_injection':'存在不可信指令内容',
        'other':'其他或不确定依据'}


def render_generation(result):
    lines=['\n\n本地生成辅助（实验）：'+result.get('status','disabled'),
           '以下覆盖整个选区的全部成员，并非仅当前成员对。',
           '模型候选不是事实或真实情感。直接统计未被修改。',
           '说明由模型选择的依据类别与消息行号组成，不显示聊天正文。']
    for p in result.get('predictions',[]):
        target=p.get('candidate_target') or '弃权'
        reason=REASON.get(p.get('reason'),'证据不足、输入受限或输出校验失败')
        evidence='、'.join(p.get('evidence_ids',[])) or '无'
        lines.append(f"{p.get('message_id') or '无ID'} → {target}；话题 {TOPIC.get(p.get('topic'),'未知')}；行为 {TYPE.get(p.get('interaction'),'未知')}\n  {reason}；证据 {evidence}（不确定）")
    return '\n'.join(lines)+'\n'
