import common.mj_helper as mjh
from common.log_helper import LOGGER
from .openai_llm import llm_client


# Natural language mapping for MJAI tiles (Chinese)
MJAI_TILE_2_NL = {
    # Man (万)
    '1m': '一万', '2m': '二万', '3m': '三万', '4m': '四万', '5mr': '红五万', '5m': '五万', '6m': '六万', '7m': '七万', '8m': '八万', '9m': '九万',
    # Pin (饼)
    '1p': '一饼', '2p': '二饼', '3p': '三饼', '4p': '四饼', '5pr': '红五饼', '5p': '五饼', '6p': '六饼', '7p': '七饼', '8p': '八饼', '9p': '九饼',
    # Sou (索)
    '1s': '一索', '2s': '二索', '3s': '三索', '4s': '四索', '5sr': '红五索', '5s': '五索', '6s': '六索', '7s': '七索', '8s': '八索', '9s': '九索',
    # Winds and dragons
    'E': '东', 'S': '南', 'W': '西', 'N': '北', 'P': '白', 'F': '发', 'C': '中',
    # Misc / placeholders
    '?': '未知'
}

DORA_DORA_MARKERS = {
    '1m': '2m', '2m': '3m', '3m': '4m', '4m': '5m', '5m': '6m', '6m': '7m', '7m': '8m', '8m': '9m', '9m': '1m',
    '1p': '2p', '2p': '3p', '3p': '4p', '4p': '5p', '5p': '6p', '6p': '7p', '7p': '8p', '8p': '9p', '9p': '1p',
    '1s': '2s', '2s': '3s', '3s': '4s', '4s': '5s', '5s': '6s', '6s': '7s', '7s': '8s', '8s': '9s', '9s': '1s',
    'E': 'S', 'S': 'W', 'W': 'N', 'N': 'E',
    'P': 'F', 'F': 'C', 'C': 'P'
}

# Common action -> NL mapping
ACTION_NL = {
    'reach': '立直', 'pon': '碰', 'chi': '吃', 'chi_low': '吃(低)', 'chi_mid': '吃(中)', 'chi_high': '吃(高)',
    'kan_select': '选择杠', 'dahai': '打牌', 'kakan': '加杠', 'daiminkan': '大明杠', 'ankan': '暗杠',
    'zimo': '自摸', 'hora': '和了', 'ryukyoku': '流局', 'nukidora': '抜きドラ', 'none': '过'
}

ACTION_NL_ADV = {
    -3: '下家', -2: '对家', -1: '上家', 0: '自己', 1: '下家', 2: '对家', 3: '上家'  # for 4-player, wrap around
}


def mjai_to_natural(tile: str) -> str:
    """Convert a single MJAI tile code to Chinese natural language.
    Falls back to the original string if unknown.
    """
    return MJAI_TILE_2_NL.get(tile, tile)


def tile_list_to_nl(tiles, tile_types) -> str:
    if not tiles:
        return '无'
    try:
        return '、'.join(f"{tile_types[i]}{mjai_to_natural(t)}" for i, t in enumerate(tiles))
    except Exception:
        return str(tiles)

def tile_list_to_nl_single(tiles) -> str:
    if not tiles:
        return '无'
    try:
        return '、'.join(mjai_to_natural(t) for t in tiles)
    except Exception:
        return str(tiles)
    
def get_dora_from_markers(dora_markers) -> list:
    """Given a list of dora markers, return the corresponding dora tiles."""
    if not dora_markers:
        return []
    dora_tiles = []
    for marker in dora_markers:
        dora = DORA_DORA_MARKERS.get(marker)
        if dora:
            dora_tiles.append(dora)
    return dora_tiles

def melds_to_nl(melds, melds_types) -> str:
    """Format melds (副露) into NL. Each meld can be a list of tile codes or a string.
    """
    if not melds:
        return '无'
    out = []
    for j, meld in enumerate(melds):
        if isinstance(meld, (list, tuple)):
            out.append(f"[{melds_types[j]}: " + '、'.join(f"{mjai_to_natural(t)}" for t in meld) + ']')
        else:
            out.append(str(meld))
    return '，'.join(out)


def parse_ai_recommendation(ai_reco: dict, is_3p: bool = False, top_k: int = 3):
    """Return up to two (action_nl, prob) pairs for the highest-probability options and a description string.

    This function prefers to parse nested `meta` formats (q_values + mask_bits). If `meta` parsing
    fails it will attempt a manual decode of q_values+mask_bits. Regardless of the original format,
    at most the top two options (by probability) are returned.
    """
    if not ai_reco:
        return [], '无'

    options = []
    info = []

    # Support nested meta dict (common pattern: ai_reco = {'type':'dahai', 'meta': {...}})
    meta_source = None
    if isinstance(ai_reco, dict) and 'meta' in ai_reco and isinstance(ai_reco['meta'], dict):
        meta_source = ai_reco['meta']
    elif isinstance(ai_reco, dict) and 'q_values' in ai_reco and 'mask_bits' in ai_reco:
        meta_source = ai_reco

    if meta_source and 'q_values' in meta_source and 'mask_bits' in meta_source:
        # Try to use helper in mj_helper first, fallback to manual parsing if needed
        try:
            options = mjh.meta_to_options(meta_source, is_3p=is_3p)
            info.append('来自 meta (q_values + mask_bits)')
        except Exception as e:
            # Fallback: manual parse using q_values and mask_bits
            try:
                mask_list = mjh.MJAI_MASK_LIST_3P if is_3p else mjh.MJAI_MASK_LIST
                q_values = meta_source.get('q_values', [])
                mask_bits = meta_source.get('mask_bits', 0)
                mask = mjh.mask_bits_to_bool_list(mask_bits)
                weight_values = mjh.softmax(q_values)
                option_list = []
                q_value_idx = 0
                for i in range(len(mask)):
                    if mask[i]:
                        name = mask_list[i] if i < len(mask_list) else f'idx_{i}'
                        w = float(weight_values[q_value_idx]) if q_value_idx < len(weight_values) else 0.0
                        option_list.append((name, w))
                        q_value_idx += 1
                option_list = sorted(option_list, key=lambda x: x[1], reverse=True)
                options = option_list
                info.append('来自 meta (手动解析 q_values + mask_bits)')
            except Exception as e2:
                options = []
                info.append(f'解析 meta 出错: {e2}')

    elif isinstance(ai_reco, dict) and 'options' in ai_reco and isinstance(ai_reco['options'], list):
        options = ai_reco['options']
        info.append('来自 options 字段')
    elif isinstance(ai_reco, dict) and 'action' in ai_reco:
        options = [(ai_reco['action'], ai_reco.get('prob'))]
        info.append('来自 action 字段')
    elif isinstance(ai_reco, dict) and 'selected' in ai_reco:
        options = [(ai_reco['selected'], ai_reco.get('prob'))]
        info.append('来自 selected 字段')
    else:
        # attempt to interpret as action->score mapping
        if isinstance(ai_reco, dict):
            try:
                options = sorted(ai_reco.items(), key=lambda kv: kv[1], reverse=True)
                info.append('从字典 (action->score) 解析')
            except Exception:
                options = []
                info.append('无法解析推荐')

    # Only convert the top two options to NL (regardless of requested top_k)
    top_n = 2
    formatted = []
    for act, prob in options[:top_n]:
        try:
            nl_act = action_to_nl(act)
        except Exception:
            nl_act = str(act)
        formatted.append((nl_act, prob))

    return formatted, '；'.join(info)


def action_to_nl(act) -> str:
    """Convert various action representations to NL-friendly string.
    Handles strings, tuples/lists (e.g. ('dahai','W')), and dicts (e.g. {'type':'dahai','pai':'W'}).

    Special case: for dahai (discard) prefer the short form '打X' (e.g. 打八饼) to match user examples.
    """
    if act is None:
        return '无'

    # tuple/list forms
    if isinstance(act, (tuple, list)):
        if len(act) >= 2 and isinstance(act[0], str) and act[0] in ACTION_NL:
            typ = act[0]
            tile = act[1] if len(act) > 1 else None
            desc = ACTION_NL.get(typ, typ)
            # short form for dahai
            if typ == 'dahai':
                if tile:
                    return f'打{mjai_to_natural(tile)}'
                return '打'
            if tile:
                return f'{desc} {mjai_to_natural(tile)}'
            return desc
        # if looks like (tile, prob) from some APIs, return tile NL
        if len(act) == 2 and isinstance(act[0], str) and isinstance(act[1], (float, int)):
            return mjai_to_natural(act[0])
        return ' '.join(str(a) for a in act)

    # dict form
    if isinstance(act, dict):
        typ = act.get('type') or act.get('action')
        pai = act.get('pai') or act.get('tile')
        if typ:
            desc = ACTION_NL.get(typ, typ)
            if typ == 'dahai':
                if pai:
                    return f'打{mjai_to_natural(pai)}'
                return '打'
            if pai:
                return f'{desc} {mjai_to_natural(pai)}'
            return desc
        if pai:
            return mjai_to_natural(pai)
        return str(act)

    # string form
    if isinstance(act, str):
        if act in ACTION_NL:
            # short form for dahai string 'dahai'
            if act == 'dahai':
                return '打'
            return ACTION_NL[act]
        return mjai_to_natural(act)

    return str(act)

def disc_type_to_nl(discard_type) -> str:
    """Convert discard type list (e.g. [True, False, ...]) to NL string indicating tsumogiri.
    """
    if not discard_type:
        return ''
    try:
        parts = []
        for idx, is_tsumogiri in enumerate(discard_type):
            if is_tsumogiri:
                parts.append(f'摸切')
            else:
                parts.append(f'手切')
        return parts
    except Exception:
        return str(discard_type)
    
def melds_info_to_nl(melds_info) -> str:
    """Convert melds info list (e.g. [('pon', 'E'), ...]) to NL string.
    """
    if not melds_info:
        return ''
    try:
        parts = []
        for info in melds_info:
            if isinstance(info, (list, tuple)) and len(info) >= 1:
                typ = info[0]
                actor = info[1] if len(info) > 1 else None
                target = info[2] if len(info) > 2 else None
                desc = ACTION_NL.get(typ)
                if target:
                    parts.append(f'{desc}（来自{ACTION_NL_ADV.get(target - actor)}）')
                else:
                    parts.append(desc)
            else:
                parts.append(str(info))
        return parts
    except Exception:
        return melds_info


def explain(game_info, kyoku_info, ai_recommendation: dict, is_3p: bool = False, top_k: int = 3) -> str:
    """Generate a concise Chinese prompt to ask an LLM to explain an AI recommendation.

    The prompt includes a short table summary, current discards/melds, the player's hand, the AI
    recommendation (top items and meta), and a short instruction for the LLM to explain the
    recommendation, list risks, propose alternatives, and give a final concise decision.
    """

    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Basic table info
    bakaze = _get(game_info, 'bakaze', '?')
    kyoku = _get(game_info, 'kyoku', '?')
    honba = _get(game_info, 'honba', 0)
    kyotaku = _get(game_info, 'kyotaku', 0)
    oya = _get(game_info, 'oya', None)
    dora_marker = _get(game_info, 'dora_marker', _get(game_info, 'dora', ['?']))
    dora = get_dora_from_markers(dora_marker)

    # Scores (prefer game_info.scores, otherwise kyoku_info.scores)
    scores = _get(game_info, 'scores', None)
    if scores is None:
        scores = _get(kyoku_info, 'scores', None)

    # My hand / draw
    my_tehai = _get(game_info, 'my_tehai', _get(game_info, 'tehai', None))
    my_tsumohai = _get(game_info, 'my_tsumohai', _get(game_info, 'tsumohai', None))

    # reach flags
    player_reached = _get(game_info, 'player_reached', _get(game_info, 'player_reach', None))

    # who is self (if provided)
    self_seat = _get(game_info, 'self_seat', _get(game_info, 'my_seat', None))

    # kyoku detail: discarded & melded
    discarded = _get(kyoku_info, 'discarded', None)
    melded = _get(kyoku_info, 'melded', None)
    discarded_type = _get(kyoku_info, 'discarded_type', None)  # if discard is tsumogiri
    melded_info = _get(kyoku_info, 'melded_info', None)          # list of (type, target) for each meld

    # normalize seat count
    seat_count = 3 if is_3p else 4
    if scores and isinstance(scores, (list, tuple)):
        seat_count = len(scores)
    elif discarded and isinstance(discarded, (list, tuple)):
        seat_count = len(discarded)
    elif melded and isinstance(melded, (list, tuple)):
        seat_count = len(melded)

    # seat names relative to oya (dealer)
    winds = ['东', '南', '西', '北']
    seat_names = []
    if oya is None:
        seat_names = [f'第{idx+1}位' for idx in range(seat_count)]
    else:
        for idx in range(seat_count):
            wind = winds[(idx - oya) % 4]
            seat_names.append(f'第{idx+1}位({wind}家)')

    bakaze_cn = {'E': '东', 'S': '南', 'W': '西', 'N': '北'}.get(bakaze, bakaze)

    # Short header
    lines = []
    lines.append('你是一个专业的日本麻将高手，擅长分析和解读麻将游戏中的策略和技巧。')
    lines.append('请基于下列牌局快照和AI推荐，简明扼要地解释AI给出概率的原因。')
    lines.append('')
    header = f'场风: {bakaze_cn}{kyoku}局；本场: {honba}本；供托: {kyotaku}；庄家: {oya+1 if oya is not None else "未知"}位；宝牌: {mjai_to_natural(tile_list_to_nl_single(dora))}。'
    if is_3p:
        header += '本局是三人麻将。'
    lines.append(header)

    # Scores
    if scores:
        score_text = '；'.join(f"{seat_names[i] if i < len(seat_names) else '第'+str(i+1)+'位'} {int(s)}分" for i, s in enumerate(scores))
        lines.append('分数: ' + score_text)
    else:
        lines.append('分数: 无')

    # Discards & melds (concise)
    lines.append('')
    lines.append('场上弃牌（按位）:')
    for i in range(seat_count):
        nm = seat_names[i] if i < len(seat_names) else f'第{i+1}位'
        nm = nm + '(我)' if i == self_seat else f'第{i+1}位'
        dis_types = disc_type_to_nl(discarded_type[i])
        melds_infos = melds_info_to_nl(melded_info[i])
        print(melds_infos, melded[i])
        disc_text = tile_list_to_nl(discarded[i], dis_types) if isinstance(discarded, (list, tuple)) and i < len(discarded) else '无'
        md_text = melds_to_nl(melded[i], melds_infos) if isinstance(melded, (list, tuple)) and i < len(melded) else '无'
        reach_flag = ''
        try:
            if isinstance(player_reached, (list, tuple)) and i < len(player_reached) and player_reached[i]:
                reach_flag = '（立直）'
        except Exception:
            reach_flag = ''
        lines.append(f'{nm}{reach_flag} 牌河: {disc_text}；副露: {md_text}')

    # My hand
    lines.append('')
    lines.append('我的手牌: ' + (tile_list_to_nl_single(my_tehai) if my_tehai else '未知'))
    lines.append('我摸到: ' + (mjai_to_natural(my_tsumohai) if my_tsumohai else '无'))

    # AI recommendation parsing
    options, info_text = parse_ai_recommendation(ai_recommendation, is_3p=is_3p, top_k=top_k)

    def _fmt_prob(p):
        if p is None:
            return ''
        try:
            pv = float(p)
        except Exception:
            return f' ({p})'
        # if value in [0,1] treat as fraction
        if 0 <= pv <= 1:
            return f' ({pv*100:.1f}%)'
        # if plausible percentage already
        if 1 <= pv <= 100:
            return f' ({pv:.1f}%)'
        return f' ({pv})'


    # Meta summary (helpful for LLM reasoning)
    meta = None
    if isinstance(ai_recommendation, dict):
        if isinstance(ai_recommendation.get('meta'), dict):
            meta = ai_recommendation.get('meta')
        elif 'q_values' in ai_recommendation or 'mask_bits' in ai_recommendation:
            meta = ai_recommendation

    if isinstance(meta, dict):
        meta_lines = []
        for k in ('shanten', 'at_furiten', 'is_greedy', 'eval_time_ns', 'batch_size'):
            if k in meta:
                meta_lines.append(f'{k}: {meta.get(k)}')
                if k == 'shanten':
                    lines.append(f'当前向听数: {meta.get(k)}')
        # q_values top-3 by value (show indices + values) if available
        qv = meta.get('q_values')
        if isinstance(qv, (list, tuple)) and len(qv) > 0:
            try:
                top_q = sorted(list(enumerate(qv)), key=lambda t: t[1], reverse=True)[:3]
                meta_lines.append('q_values(top3 indices->value): ' + ', '.join(f'{i}->{v:.4f}' for i, v in top_q))
            except Exception:
                pass
        # if meta_lines:
        #     lines.append('')
        #     lines.append('AI meta: ' + '；'.join(meta_lines))

    lines.append('')
    lines.append(f'AI 推荐:')
    if options:
        for idx, (act, prob) in enumerate(options[:top_k]):
            lines.append(f'{idx+1}. {action_to_nl(act)}{_fmt_prob(prob)}')
    else:
        # fallback: attempt to show top-level ai_recommendation fields
        if isinstance(ai_recommendation, dict) and 'type' in ai_recommendation:
            desc = action_to_nl(ai_recommendation)
            prob = ai_recommendation.get('prob')
            lines.append('1. ' + desc + _fmt_prob(prob))
        else:
            lines.append('无可解析的推荐')
    # Instructions for the LLM (concise)
    lines.append('')
    lines.append("""核心原则​​：何切是风险与回报的动态权衡，需同步评估「手牌潜力」「当前局势」「对手信息」「风格偏好」，而非依赖单一标准。
                 
你可以参考以下几个方面来思考为什么AI会推荐打掉这些牌：
1. 牌效与进张：考虑打掉的牌是否提升了手牌的整体效率和进张机会，是否会降低向听数。
2. 役种与打点：评估打掉的牌是否有助于达成特定的役种，是否影响最终的打点。
3. 安全性与防守：分析打掉的牌是否是安全牌，是否有助于降低放铳风险。注意：这个考量往往在对手立直或局势紧张时更为重要，在早巡一般不会考虑。
4. 防守与对攻：如果对手已经立直，评估打掉的牌是否安全，是否可能被对手和了。需要考虑自己牌的价值。如果自己的牌很大且已经听牌，可能会选择冒险打出一些不太安全的牌以追求和牌；如果自己的牌较小且未听牌，可能会更倾向于打出安全牌以防止放铳。
5. 听牌质量：如果打掉的牌能让手牌进入听牌状态，评估听牌的质量（如听牌数、役种、打点）。
6. 局势与对手：结合当前的局势（如分数、剩余牌数、对手状态）来判断打掉的牌是否符合整体战略。如果处于领先位置，可能更倾向于保守打法；如果落后，可能需要冒险追求高打点。
7. 立直与默听：如果听牌且无役，且局势需要打点，一般会选择立直。但如果听的太差（如山里最多只有一枚），可以等待改良。如果听牌有役，且局势允许，可以选择默听以追求更高和率。
8. 和牌与见逃：如果已经听牌，如果是All Last，需要评估是否荣和牌或自摸后能逆转顺位或避四。如果不是All Last，一般会选择和牌以结束局面，除非确定可以拿到更多的流局听牌罚符。

对于AI给出的若干个推荐打掉的牌或做出的鸣/荣/自摸决定，请你：
对于每个动作，给出一个词概括AI为什么决定这样做。然后用一句简短的话具体分析打出/鸣/和这张牌的好处和坏处。
请用类似于以下的格式输出：
牌：原因；具体分析。
                 
其中牌为AI推荐打出的牌，原因是个词，具体分析为一句简短的话具体分析打出这张牌的好处和坏处。
如：
九万：牌效；早巡九万孤张且进张差，打出可提升整体牌效。
碰：速攻；碰牌加速和牌进程，由于处于领先位置，所以可以快速过庄。
""")

    prompt = '\n'.join(lines)
    LOGGER.info("生成的提示语: %s", prompt)
    explanation = llm_client.send_request("", prompt)
    LOGGER.info("生成的解释: %s", explanation)
    return explanation
