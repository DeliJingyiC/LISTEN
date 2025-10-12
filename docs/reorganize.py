#!/usr/bin/env python3
import re

# Read the HTML file
with open('index.html', 'r') as f:
    content = f.read()

# Find the examples section (it comes before experiments currently)
examples_match = re.search(r'(<section id="examples".*?</section>)', content, re.DOTALL)
experiments_match = re.search(r'(<section id="experiments".*?</section>)', content, re.DOTALL)

if examples_match and experiments_match:
    examples_section = examples_match.group(1)
    
    # Remove examples section from its current position
    content = content.replace(examples_section, '', 1)
    
    # Find where experiments section ends (after removing examples)
    experiments_match = re.search(r'(<section id="experiments".*?</section>)', content, re.DOTALL)
    
    if experiments_match:
        # Insert examples after experiments
        insert_pos = experiments_match.end()
        content = content[:insert_pos] + '\n\n    ' + examples_section + content[insert_pos:]
        
        # Write back
        with open('index.html', 'w') as f:
            f.write(content)
        
        print("✓ Successfully moved examples section after experiments section")
    else:
        print("✗ Could not find experiments section")
else:
    print("✗ Could not find examples or experiments sections")

