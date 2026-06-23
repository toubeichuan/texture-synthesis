PALETTE = {
    0: [255, 255, 255], # white  -  background
    1: [204, 50, 50],   # red    -  old
    2: [231, 180, 22],  # yellow -  update
    3: [45, 201, 55]    # green  -  new
}

QUAD_WEIGHTS = {
    0: 0, # background
    1: 0.1,   # old
    2: 0.5,  # update
    3: 1    # new
}

VIEWPOINTS = {
    1: {
        "azim": [
            0
        ],
        "elev": [
            0
        ],
        "sector": [
            "front"
        ]
    },
    2: {
        "azim": [
            0,
            30
        ],
        "elev": [
            0,
            0
        ],
        "sector": [
            "front",
            "front"
        ]
    },
    4: {
        "azim": [
            45,
            315,
            135,
            225,
        ],
        "elev": [
            0,
            0,
            0,
            0,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",
        ]
    },
    6: {
        "azim": [
            0,
            90,
            270,
            0,
            180,
            0
        ],
        "elev": [
            0,
            0,
            0,
            90,
            0,
            -90
        ],
        "sector": [
            "front",
            "right",
            "left",
            "top",
            "back",
            "bottom",
        ]
    },
    "shapenet": {
        "azim": [
            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
            0,
            0
        ],
        "elev": [
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            90,
            -90
        ],
        "sector": [
            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
            "top",
            "bottom",
        ]
    },
    "labubu18": {
        "azim": [
            0, 45, 315, 90, 270, 135, 225, 180,
            45, 135, 225, 315,
            45, 135, 225, 315,
            0, 0,
        ],
        "elev": [
            15, 15, 15, 15, 15, 15, 15, 15,
            45, 45, 45, 45,
            -30, -30, -30, -30,
            90, -90,
        ],
        "sector": [
            "front", "front right", "front left", "right",
            "left", "back right", "back left", "back",
            "upper front right", "upper back right",
            "upper back left", "upper front left",
            "lower front right", "lower back right",
            "lower back left", "lower front left",
            "top", "bottom",
        ],
    },
    "objaverse": {
    "azim": [
        # Phase A (36) : elev=15, azim scan (mode 2 expanded)
        0, 5, 355, 10, 350, 15, 345, 20,
        340, 25, 335, 30, 330, 35, 325, 40,
        320, 45, 315, 50, 310, 55, 305, 60,
        300, 65, 295, 70, 290, 75, 285, 80,
        280, 85, 275, 90,

        # Phase B (15): azim=20, elev ramp up 20->90 (step +5)
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,

        # Phase C (3): elev=90, azim 15->10->5 (step -5)
        15, 10, 5,

        # Phase D (8): azim=5, key elevs (8)
        5, 5, 5, 5, 5, 5, 5, 5,

        # Phase E (10): elev=-90, azim scan 10->55 (step +5)
        10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
    ],

    "elev": [
        # Phase A (36): elev=15
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15,

        # Phase B (15): elev 20->90
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,

        # Phase C (3): elev=90
        90, 90, 90,

        # Phase D (8): key elevs (8)
        85, 60, 35, 10, -15, -40, -65, -90,

        # Phase E (10): elev=-90
        -90, -90, -90, -90, -90, -90, -90, -90, -90, -90,
    ],

    "sector": [
        # =========================
        # Phase A (36): elev=15, descriptive text
        # =========================
        "front | elev 15°",
        "front slight right | elev 15°",
        "front slight left | elev 15°",
        "front-right | elev 15°",
        "front-left | elev 15°",
        "front-right (more) | elev 15°",
        "front-left (more) | elev 15°",
        "right-front | elev 15°",

        "right-front (more) | elev 15°",
        "right-front (more) | elev 15°",
        "right-front (more) | elev 15°",
        "right | elev 15°",
        "right | elev 15°",
        "right | elev 15°",
        "right | elev 15°",
        "right | elev 15°",

        "right | elev 15°",
        "right-back (slight) | elev 15°",
        "right-back | elev 15°",
        "right-back | elev 15°",
        "right-back | elev 15°",
        "right-back | elev 15°",
        "right-back | elev 15°",
        "right-back | elev 15°",

        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",
        "back-right | elev 15°",

        "back-right (more) | elev 15°",
        "back-right (more) | elev 15°",
        "back-right (more) | elev 15°",
        "right-back (towards side) | elev 15°",

        # =========================
        # Phase B (15): azim=20, elev 20->90
        # =========================
        "front-right | elev 20°",
        "front-right | elev 25°",
        "front-right | elev 30°",
        "front-right | elev 35°",
        "front-right | elev 40°",
        "front-right | elev 45°",
        "front-right | elev 50°",
        "front-right | elev 55°",
        "front-right | elev 60°",
        "front-right | elev 65°",
        "front-right | elev 70°",
        "near top-front-right | elev 75°",
        "near top-front-right | elev 80°",
        "near top-front-right | elev 85°",
        "top-front-right | elev 90°",

        # =========================
        # Phase C (3): elev=90, azim shift
        # =========================
        "top-front-right | elev 90°",
        "top-front-right | elev 90°",
        "top-front-right | elev 90°",

        # =========================
        # Phase D (8): azim=5, key elevs
        # =========================
        "front slight right | elev 85°",
        "front slight right | elev 60°",
        "front slight right | elev 35°",
        "front slight right | elev 10°",
        "front slight right | elev -15°",
        "front slight right | elev -40°",
        "near bottom | elev -65°",
        "bottom | elev -90°",

        # =========================
        # Phase E (10): bottom azim scan
        # =========================
        "bottom-front-right | elev -90°",
        "bottom-front-right | elev -90°",
        "bottom-front-right | elev -90°",
        "bottom-right-ish | elev -90°",
        "bottom-right | elev -90°",
        "bottom-right | elev -90°",
        "bottom-right | elev -90°",
        "bottom-right | elev -90°",
        "bottom-right | elev -90°",
        "bottom-right | elev -90°",
        ]
    },
    12: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
        ]
    },
    20: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
        ]
    },
    36: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,

            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "top front",
            "top right",
            "top left",
            "top right",
            "top left",
            "top right",
            "top left",
            "top back",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",
        ]
    },
    68: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,

            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,

            -30,
            -30,
            -30,
            -30,
            -30,
            -30,
            -30,
            -30,

            -60,
            -60,
            -60,
            -60,
            -60,
            -60,
            -60,
            -60,

            -15,
            -15,
            -15,
            -15,
            -15,
            -15,
            -15,
            -15,

            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "top front",
            "top right",
            "top left",
            "top right",
            "top left",
            "top right",
            "top left",
            "top back",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "bottom front",
            "bottom right",
            "bottom left",
            "bottom right",
            "bottom left",
            "bottom right",
            "bottom left",
            "bottom back",

            "bottom front right",
            "bottom front left",
            "bottom front right",
            "bottom front left",
            "bottom back right",
            "bottom back left",
            "bottom back right",
            "bottom back left",

            "bottom front right",
            "bottom front left",
            "bottom front right",
            "bottom front left",
            "bottom back right",
            "bottom back left",
            "bottom back right",
            "bottom back left",
        ]
    }
}
