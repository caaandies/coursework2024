from argparse import ArgumentParser
from random import choice, choices

parser = ArgumentParser()

parser.add_argument('--n_prompts', type=int, default=4000)

parser.add_argument('--file_name', type=str, default='fclips/datasets/prompt_dataset')

args = parser.parse_args()

keywords = {
    'colors': [
        'White', 'Black', 'Green', 'Yellow', 'Pink', 'Purple', 'Blue', 'Orange', 'Red',
        'Violet', 'Silver', 'Gray', 'Maroon', 'Teal', 'Aqua', 'Beige', 'Brown', 'Crimson',
        'Cyan', 'Gold', 'Greenyellow', 'Hotpink', 'Khaki', 'Magenta', 'Turquoise',
        'Fuchsia', 'Salmon', 'Seashell', 'Chocolate', 'Peru', 'Whitesmoke', 'Honeydew',
        'Rosybrown', 'Saddlebrown', 'Seagreen', 'Slategray', 'Steelblue', 'Indianred',
        'Olive', 'Lime', 'Navy', 'Lavender', 'Indigo', 'Ivory'
        ],
    'objects': [
        'Wool', 'Color pencil', 'Pencil', 'Brush', 'Color brush', 'Crystals',
        'Color painting', 'Crayon', 'Lines', 'Watercolor', 'Stone wall',
        'Cloud', 'Underwater', 'Fire', 'Metal', 'Lightning', 'Wave', 'Flames', 'Leafy',
        'Grassy', 'Darkness', 'Wooden', 'Snow', 'Iceberg', 'Cartoon', 'Comic', 'Squares',
        'Lava', 'Vines', 'Magma', 'Desert sand', 'Water waves', 'Lace'
    ],
    'art styles': [
        'Acrylic', 'Oil Painting', 'Mosaic', 'Cubism', 'Monet'
    ],
    'Textures': [
        'Crystalline', 'Cracked', 'Crosshatched', 'Fibrous', 'Freckled', 'Grid',
        'Honeycombed', 'Meshed', 'Perforated', 'Porous', 'Scaly', 'Smeared', 'Studded',
        'Swirly', 'Veined', 'Waffled', 'Woven', 'Wrinkled', 'Pleated', 'Sprinkled', 'Knitted'
    ]
}

prefixes = ["", "style of ", "a pixelated photo of ", "a blurry photo of ", "a sketch of ", "a black and white photo "]
suffixes = ["", " style", " style painting"]

prompts = set()

while len(prompts) != args.n_prompts:
    color = choice(keywords['colors']) + " color"
    object = choice(keywords['objects'])
    art_style = choice(keywords['art styles'])
    texture = choice(keywords['Textures'])
    prefix = choices(prefixes, weights=[5, 1, 1, 1, 1, 1])[0]
    suffix = choices(suffixes, weights=[2, 1, 1])[0]

    combinations = [
        f"{color} {texture}",
        f"{color} {art_style}",
        f"{color} {object}",
        f"{art_style} {texture}",
        f"{art_style} {object}"
    ]

    prompt = choice(combinations)

    augment = [
        f"{prefix}{prompt}",
        f"{prompt}{suffix}"
    ]

    prompt = choice(augment) + '\n'
    prompts.add(prompt)

with open(args.file_name, 'w') as file:
    for prompt in prompts:
        file.write(prompt)
