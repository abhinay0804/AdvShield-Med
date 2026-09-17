import json

transcript_path = '/home/abhi/.gemini/antigravity/brain/03524d43-8df1-4328-be87-5b97ceae31ec/.system_generated/logs/transcript_full.jsonl'
blueprint_path = '/mnt/shared/Projects/PJT/advshield-med/documentation/project_blueprint.md'

target_content = ""
with open(transcript_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'VIEW_FILE' and 'implementation_plan.md' in data.get('content', ''):
            content = data['content']
            # Parse the VIEW_FILE output to get the raw text
            # Format is:
            # Created At: ...
            # Completed At: ...
            # File Path: ...
            # Total Lines: ...
            # Total Bytes: ...
            # Showing lines 1 to 358
            # The following code has been modified...
            # 1: # AdvShield-Med...
            
            lines = content.split('\n')
            raw_lines = []
            parsing = False
            for l in lines:
                if l.startswith('1: '):
                    parsing = True
                if parsing:
                    if ': ' in l:
                        raw_lines.append(l.split(': ', 1)[1])
            target_content = '\n'.join(raw_lines)

# One last check to make sure we don't save the "The above content shows the entire..." footer
if target_content:
    target_content = target_content.split('The above content shows the entire, complete file')[0].strip()
    
    # Save it to the documentation folder
    import os
    os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
    with open(blueprint_path, 'w') as out:
        out.write(target_content)
    print("Successfully recovered blueprint to", blueprint_path)
else:
    print("Could not find the blueprint in transcript_full.jsonl")
