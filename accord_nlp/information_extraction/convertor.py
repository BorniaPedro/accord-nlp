# Created By Pouyan Parsafard - 8/24/2023
# Modified by Hansi on 28/08/2023
# Updated for robust index parsing and ontological constraint filtering

import re
from collections import Counter
from itertools import combinations

import graphviz
import pandas as pd
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree


def entity_pairing(sentence, entities):
    """
    Pairs detected entities into candidate relation instances with boundary safeguards.

    :param sentence: str, input sentence
    :param entities: list of {entity text: tag}
        e.g. [{'Perimeter': 'B-object'}, {'insulation': 'I-object'}, {'should': 'O'}, {'be': 'O'}, {'continuous': 'B-quality'}]
    :return: pd.DataFrame with formatted pair candidates
    """
    df_output = pd.DataFrame(columns=['sentence', 'format1', 'output', '<e1>', '<e2>', '<e1>_tag', '<e2>_tag',
                                      '<e1>_startIndex', '<e1>_endIndex', '<e2>_startIndex', '<e2>_endIndex',
                                      'prediction'])

    if not entities or len(entities) == 0:
        return df_output

    tokens = [list(dict(x).keys())[0] for x in entities]
    tags = [list(dict(x).values())[0] for x in entities]

    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    original_text = []
    for subtree in ne_tree:
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))

    if len(original_text) < 2:
        return df_output

    none_entityType_text = [x[0] for x in original_text]
    c = Counter()
    c.update(Counter(none_entityType_text))
    counting_noneEntityType_text = []

    for k, v in c.most_common():
        temp_list = [x for x in original_text if x[0] == k]
        if v == 1:
            for x in original_text:
                if x[0] == k:
                    counting_noneEntityType_text.append((k, 0, x[1]))
        else:
            for j in range(1, v + 1):
                z = (k, j, temp_list[j - 1][1])
                counting_noneEntityType_text.append(z)

    combinations_list = list(combinations(counting_noneEntityType_text, 2))

    # Build entity occurrence index dictionary protected with re.escape
    entity_index_dictionary = {}
    for term in set(none_entityType_text):
        result = list(re.finditer(re.escape(term), sentence))
        entity_index_dictionary[term] = {}
        if len([x for x in none_entityType_text if x == term]) > 1:
            counter_below = 1
        else:
            counter_below = 0

        for match in result:
            entity_index_dictionary[term][counter_below] = match.span()
            counter_below += 1

        # Fallback safeguard in case key 0 was not populated
        if 0 not in entity_index_dictionary[term] and len(result) > 0:
            entity_index_dictionary[term][0] = result[0].span()

    combinations_list_withindex = []
    for x in combinations_list:
        try:
            # Safe boundary retrieval with defensive fallback
            span_e1 = entity_index_dictionary[x[0][0]].get(x[0][1], next(iter(entity_index_dictionary[x[0][0]].values()), None))
            span_e2 = entity_index_dictionary[x[1][0]].get(x[1][1], next(iter(entity_index_dictionary[x[1][0]].values()), None))

            if span_e1 is None or span_e2 is None:
                continue

            y0 = list(x[0]) + [span_e1]
            y1 = list(x[1]) + [span_e2]

            # Insert <e1> and <e2> tags around entities
            part1 = sentence[: y0[3][0]]
            part2 = sentence[y0[3][1]: y1[3][0]]
            part3 = sentence[y1[3][1]:]
            y2 = part1 + '<e1>' + y0[0] + '</e1>' + part2 + '<e2>' + y1[0] + '</e2>' + part3

            combinations_list_withindex.append((tuple(y0), tuple(y1), y2))
        except (KeyError, IndexError):
            continue

    df_output_rowNumber = 0
    for x in combinations_list_withindex:
        df_output.loc[df_output_rowNumber, 'sentence'] = sentence
        df_output.loc[df_output_rowNumber, 'format1'] = entities
        df_output.loc[df_output_rowNumber, 'output'] = x[2]
        df_output.loc[df_output_rowNumber, '<e1>'] = x[0][0]
        df_output.loc[df_output_rowNumber, '<e2>'] = x[1][0]
        df_output.loc[df_output_rowNumber, '<e1>_tag'] = x[0][2]
        df_output.loc[df_output_rowNumber, '<e2>_tag'] = x[1][2]
        df_output.loc[df_output_rowNumber, '<e1>_startIndex'] = x[0][3][0]
        df_output.loc[df_output_rowNumber, '<e1>_endIndex'] = x[0][3][1]
        df_output.loc[df_output_rowNumber, '<e2>_startIndex'] = x[1][3][0]
        df_output.loc[df_output_rowNumber, '<e2>_endIndex'] = x[1][3][1]
        df_output_rowNumber += 1

    return df_output


def apply_ontological_filter(df):
    """
    Filters out relation predictions that violate CODE-ACCORD domain ontology constraints.
    Incompatible semantic pairings are mapped to the null class ('none').

    :param df: pd.DataFrame containing paired entities and RE predictions
    :return: pd.DataFrame with updated predictions
    """
    if df is None or len(df) == 0:
        return df

    comp_relations = {'equal', 'greater', 'greater-equal', 'less', 'less-equal'}
    filtered_preds = []

    for _, row in df.iterrows():
        pred = row.get('prediction', 'none')
        if pred == 'none' or pd.isna(pred):
            filtered_preds.append('none')
            continue

        t1 = str(row['<e1>_tag']).lower().replace('b-', '').replace('i-', '')
        t2 = str(row['<e2>_tag']).lower().replace('b-', '').replace('i-', '')

        # Two numerical values never hold a direct normative relation between each other
        if t1 == 'value' and t2 == 'value':
            filtered_preds.append('none')
            continue

        # Comparative relations require at least one participating entity to be a 'value'
        if pred in comp_relations and (t1 != 'value' and t2 != 'value'):
            filtered_preds.append('none')
            continue

        # 'part-of' represents physical/structural composition and never involves values
        if pred == 'part-of' and (t1 == 'value' or t2 == 'value'):
            filtered_preds.append('none')
            continue

        filtered_preds.append(pred)

    df['prediction'] = filtered_preds
    return df


def graph_building(df, view=True):
    """
    Builds Graphviz dependency graphs based on active entity pairs and relations.

    :param df: pd.DataFrame containing paired entities and RE predictions
    :param view: bool, whether to immediately open the generated image
    :return: graphviz.Graph instance
    """
    dict_entityColors = {'object': '#dae8fc', 'quality': '#e1d5e7', 'property': '#fff2cc', 'value': '#ffffff'}
    graph = graphviz.Graph(format='png')

    new_df = df[df['prediction'] != 'none'].reset_index(drop=True)
    new_df = new_df[new_df['prediction'].notnull()].reset_index(drop=True)

    for i in range(len(new_df)):
        nodeA = f"{new_df.loc[i, '<e1>']}\n|<{new_df.loc[i, '<e1>_tag']}>\n|{new_df.loc[i, '<e1>_startIndex']}\n|{new_df.loc[i, 'sentence']}"
        nodeB = f"{new_df.loc[i, '<e2>']}\n|<{new_df.loc[i, '<e2>_tag']}>\n|{new_df.loc[i, '<e2>_startIndex']}\n|{new_df.loc[i, 'sentence']}"

        splitted_nodeA = nodeA.split('|')
        splitted_nodeB = nodeB.split('|')

        tag_e1 = str(new_df.loc[i, '<e1>_tag']).lower().replace('b-', '').replace('i-', '')
        tag_e2 = str(new_df.loc[i, '<e2>_tag']).lower().replace('b-', '').replace('i-', '')

        graph.node(nodeA, label=splitted_nodeA[0] + splitted_nodeA[1], style='filled',
                   fillcolor=dict_entityColors.get(tag_e1, '#ffffff'))
        graph.node(nodeB, label=splitted_nodeB[0] + splitted_nodeB[1], style='filled',
                   fillcolor=dict_entityColors.get(tag_e2, '#ffffff'))
        graph.edge(nodeA, nodeB, label=str(new_df.loc[i, 'prediction']))

    if view:
        graph.render(view=view)

    return graph