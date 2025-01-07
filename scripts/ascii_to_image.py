"""
Convert ASCII art to image.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def ascii_to_image(ascii_file, output_file, font_size=24):
    # Read ASCII art
    with open(ascii_file, 'r') as f:
        lines = f.readlines()
    
    # Calculate image size with wider character spacing
    char_width = font_size * 0.7  # Wider spacing for characters
    line_height = font_size + 8
    width = max(len(line) for line in lines) * int(char_width)
    height = len(lines) * line_height
    
    # Create image with some padding
    padding = 40  # Increased padding
    image = Image.new('RGB', (width + 2*padding, height + 2*padding), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a monospace font
        font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text
    y = padding
    for line in lines:
        draw.text((padding, y), line.rstrip('\n'), fill='black', font=font)
        y += line_height
    
    # Save image
    image.save(output_file)

if __name__ == "__main__":
    # Create scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)
    
    # Convert ASCII art to image with larger font size
    ascii_to_image(
        "assets/model_architecture.txt",
        "assets/model_architecture.png",
        font_size=28  # Increased font size further
    ) 