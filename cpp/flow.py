import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_sd_card_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    color_write = '#e1f5fe'
    color_read = '#fff3e0' 
    color_problem = '#ffebee'
    color_solution = '#e8f5e8'
    
    # Title
    ax.text(5, 11.5, 'SD Card Addressing Mismatch Flowchart', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # === PROBLEM FLOW ===
    ax.text(2.5, 10.8, 'PROBLEM: Dual Addressing Schemes', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    
    # Write System Flow
    ax.text(1.5, 10.2, 'WRITE SYSTEM', fontweight='bold', ha='center')
    write_flow = [
        (1.5, 9.8, 'Subsystem Data', color_write),
        (1.5, 9.2, 'Central Buffer', color_write),
        (1.5, 8.6, 'Calculate: Partition Start + sd_offsets.xxx', color_write),
        (1.5, 8.0, f'Write to: EVTLOG_START + {25}', color_write),  # Example offset
        (1.5, 7.4, 'Data Stored ✓', color_write)
    ]
    
    for x, y, text, color in write_flow:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.2), 2.4, 0.4, 
                                   boxstyle="round,pad=0.1", facecolor=color))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # Read System Flow  
    ax.text(4.0, 10.2, 'READ SYSTEM', fontweight='bold', ha='center')
    read_flow = [
        (4.0, 9.8, 'Command: set_playback_range()', color_read),
        (4.0, 9.2, 'Absolute Address: 0x1000', color_read),
        (4.0, 8.6, 'transmit_sector(0x1000)', color_read),
        (4.0, 8.0, 'Read from: Sector 0x1000', color_read),
        (4.0, 7.4, 'Garbage Data ✗', color_problem)
    ]
    
    for x, y, text, color in read_flow:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.2), 2.4, 0.4, 
                                   boxstyle="round,pad=0.1", facecolor=color))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # Problem Connection
    ax.annotate('Address Mismatch!', xy=(2.8, 7.8), xytext=(3.2, 6.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # === SOLUTION FLOW ===
    ax.text(7.5, 10.8, 'SOLUTION: Translation Layer', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='green')
    
    solution_flow = [
        (7.5, 9.8, 'Command: set_playback_range()', color_read),
        (7.5, 9.2, 'Absolute Address: 0x1000', color_read),
        (7.5, 8.6, 'Translation Layer', color_solution),
        (7.5, 8.0, 'Convert: 0x1000 → EVTLOG_START + 25', color_solution),
        (7.5, 7.4, 'Read Correct Data ✓', color_solution)
    ]
    
    for x, y, text, color in solution_flow:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.2), 2.4, 0.4, 
                                   boxstyle="round,pad=0.1", facecolor=color))
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    # Connection to write system
    ax.plot([7.5, 5.5], [7.8, 7.8], 'g--', lw=2)
    ax.plot([5.5, 1.5], [7.8, 7.8], 'g--', lw=2)
    ax.text(4.5, 7.9, 'Same Physical Location', 
            ha='center', va='bottom', fontsize=8, color='green')
    
    # SD Card Visualization
    ax.add_patch(patches.Rectangle((0.5, 5.5), 8, 1.2, facecolor='#f5f5f5', edgecolor='black'))
    ax.text(4.5, 6.1, 'SD CARD PHYSICAL LAYOUT', ha='center', va='center', fontweight='bold')
    
    # Partitions
    partitions = [
        (1.0, 5.7, 1.5, 0.3, 'SYSINFO', '#bbdefb'),
        (2.6, 5.7, 2.0, 0.3, 'EVTLOG', '#c8e6c9'),
        (4.7, 5.7, 1.8, 0.3, 'HK DATA', '#fff9c4'),
        (6.6, 5.7, 1.8, 0.3, 'XACT', '#ffcdd2')
    ]
    
    for x, y, width, height, label, color in partitions:
        ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='black'))
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=8)
    
    # Write position marker
    ax.plot([2.6 + 0.3, 2.6 + 0.3], [6.0, 5.7], 'r-', lw=2)
    ax.text(2.6 + 0.3, 6.05, 'Actual Write\nPosition', ha='center', va='bottom', fontsize=8, color='red')
    
    # Read position marker (wrong)
    ax.plot([3.5, 3.5], [6.0, 5.7], 'b-', lw=2)
    ax.text(3.5, 6.05, 'Command Read\nPosition', ha='center', va='bottom', fontsize=8, color='blue')
    
    plt.tight_layout()
    plt.show()

# Generate the flowchart
create_sd_card_flowchart()