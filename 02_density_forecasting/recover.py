import json
import os

log_path = r'C:\Users\fe_ma\.gemini\antigravity\brain\b2fd523a-a91d-4bc8-95c2-0a7f421f62e6\.system_generated\logs\transcript.jsonl'
target = '05_vix_baseline_vs_ml.ipynb'

found = False
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            step = json.loads(line)
        except:
            continue
            
        content = step.get('content', '')
        if step.get('type') in ('TOOL_RESPONSE', 'SYSTEM') or 'tool_calls' in step:
            tool_calls = step.get('tool_calls', [])
            if tool_calls:
                for tc in tool_calls:
                    if tc.get('tool_response'):
                        content += tc['tool_response'].get('output', '')
            
            if 'The following code has been modified to include a line number' in content and target in content:
                print("FOUND IT! Length:", len(content))
                found = True
                recovered_lines = []
                for cl in content.split('\n'):
                    if ': ' in cl:
                        parts = cl.split(': ', 1)
                        if parts[0].isdigit():
                            recovered_lines.append(parts[1])
                
                with open(r'C:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\recovered.ipynb', 'w', encoding='utf-8') as out:
                    out.write('\n'.join(recovered_lines))
                print("Saved to recovered.ipynb")

if not found:
    print("Not found in transcript.")
