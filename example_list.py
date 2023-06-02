example_1 = [
    "LAION-Glyph-10M-Epoch-6",
    "A gift card with text ""Happy Birthday"" and roses on it.",
    "Happy Birthday", 0.47, 0, 0.24, 0.4, 5, 1,
    "", 0.3, 0, 0.15, 0.15, 0, 1,
    "", 0.3, 0, 0.15, 0.65, 0, 1,
    "", 0.3, 0, 0.5, 0.65, 0, 1,
    5,512,20,False,1,9,0,0,
    "4K, dslr, best quality, extremely detailed",
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
]
# teaser examples in the report (updating...)
# only could generate similar examples due to the fact that our released checkpoints are different from the checkpoint used in the original report.
example_2 = [
    "LAION-Glyph-10M-Epoch-6",
    'Newspaper with the headline "Aliens Found in Space" and "Monster Attacks Mars".',
    'Aliens Found in Space', 0.8, 0, 0.1, 0.1, 0, 1,
    'Monster Attacks Mars', 0.8, 0, 0.1, 0.45, 0, 1,
    "", 0.3, 0, 0.15, 0.65, 0, 1,
    "", 0.3, 0, 0.5, 0.65, 0, 1,
    5,512,20,False,1,9,430637146, 
    0, "best quality, extremely detailed", #"4K, dslr, best quality, extremely detailed",
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
]
examples = [example_1, example_2]

# example_3 = [
#     "LAION-Glyph-10M-Epoch-6",
#     'A decorative greeting card that reads "Congratulations on achieving state of the art".',
#     'Congratulations', 0.6, 0, 0.2, 0.1, 0, 1,
#     'on achieving', 0.5, 0, 0.25, 0.22, 0, 1,
#     'state of the art', 0.6, 0, 0.21, 0.34, 0, 1,
#     "", 0.3, 0, 0.5, 0.65, 0, 1,
#     5,512,20,False,1,9, 1540281202, #364285590,
#     0, "best quality, extremely detailed",
#     "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# ]
# examples = [example_1, example_2, example_3]