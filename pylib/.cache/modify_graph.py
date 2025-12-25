import json
import os

file_path = 'current_graph.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Find subloop node
subloop = None
for node in data['nodes']:
    if node['id'] == 'subloop_node':
        subloop = node
        break

if not subloop:
    print("Error: subloop_node not found")
    exit(1)

sub_graph = subloop['config']['sub_graph']
nodes = sub_graph['nodes']
edges = sub_graph['edges']

# 1. Create New Nodes
new_nodes = [
    {
      "id": "step3_init_vars",
      "type": "python_script",
      "label": "3. Init Correction Vars",
      "position": { "x": 450, "y": 170 },
      "config": {
        "code": "context['correction_trials'] = 0"
      }
    },
    {
      "id": "step5_1_check_retry",
      "type": "condition_code",
      "label": "5.1 Retry?",
      "position": { "x": 1350, "y": 130 },
      "config": {
        "code": "result = (not context.get('is_improved', False)) and (context.get('correction_trials', 0) < 5)"
      }
    },
    {
      "id": "step5_2_llm_fix",
      "type": "llm_generate",
      "label": "5.2 LLM Correction",
      "position": { "x": 1350, "y": -50 },
      "config": {
        "model": "flash",
        "timeout": 600,
        "session_mode": "inherit",
        "session_id_input": "session_id",
        "response_output": "correction_log",
        "user_template": "{DEFAULT_SYS}\n\nThe previous attempt failed to improve the metric.\nCurrent Metric: {current_metric}\nParent Metric: {parent_metric}\nCorrection Trial: {correction_trials}/5\n\nEvaluator Output:\n{test_output}\n\nTask:\nAnalyze the failure and the evaluator output.\nModify 'strategy.py' to fix the issue or try a different approach to improve the metric.\nEnsure the code is valid and runnable."
      }
    },
    {
      "id": "step5_3_inc_trial",
      "type": "python_script",
      "label": "5.3 Inc Trial",
      "position": { "x": 1100, "y": -50 },
      "config": {
        "code": "context['correction_trials'] = context.get('correction_trials', 0) + 1"
      }
    }
]

# Check if nodes already exist to avoid duplicates
existing_ids = set(n['id'] for n in nodes)
for n in new_nodes:
    if n['id'] not in existing_ids:
        nodes.append(n)

# 2. Modify Edges

# Edge 1: n_1766207420852 -> step4_eval  ==> n_1766207420852 -> step3_init_vars
edge_found = False
for edge in edges:
    if edge['source'] == 'n_1766207420852' and edge['target'] == 'step4_eval':
        edge['target'] = 'step3_init_vars'
        edge_found = True
        break
if not edge_found:
    # If edge not found, create it (maybe structure is different?)
    pass

# Edge 2: step5_check -> step6_save_metric ==> step5_check -> step5_1_check_retry
for edge in edges:
    if edge['source'] == 'step5_check' and edge['target'] == 'step6_save_metric':
        edge['target'] = 'step5_1_check_retry'
        break

# 3. Add New Edges
new_edges = [
    {"source": "step3_init_vars", "target": "step4_eval"},
    {"source": "step5_1_check_retry", "target": "step5_2_llm_fix", "label": "true"},
    {"source": "step5_1_check_retry", "target": "step6_save_metric", "label": "false"},
    {"source": "step5_2_llm_fix", "target": "step5_3_inc_trial"},
    {"source": "step5_3_inc_trial", "target": "step4_eval"}
]

# Avoid duplicates in edges
# We check roughly by source/target
existing_edge_keys = set((e['source'], e['target']) for e in edges)

for ne in new_edges:
    if (ne['source'], ne['target']) not in existing_edge_keys:
        edges.append(ne)

# Write back
with open(file_path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Graph updated successfully.")