import logging
import unittest
import numpy

import seglinreg


logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)


class SegLinRegTestCase(unittest.TestCase):
    def test_something(self):
        obj = seglinreg.SegLinRegAuto(int(numpy.random.sample() * 10 + 2))

        normal = numpy.random.standard_normal(1 + numpy.random.sample() * 1000)

        data = []
        n = 0
        for val in normal:
            n += numpy.random.sample()
            data.append((n, val))

        chunks = obj.calculate(data)
        logging.info("Result: %s", chunks)
        res = [val for val in chunks.get_regression_data()]

        self.assertEqual(len(res), len(data))


    def test_real1(self):
        test1 = [(0, 506224.0), (1, 535982.0), (2, 534981.0), (3, 542664.0), (4, 547944.0), (5, 541522.0),
                 (6, 536643.0), (7, 536203.0), (8, 521291.0), (9, 510602.0), (10, 508576.0), (11, 496740.0),
                 (12, 495249.0), (13, 491789.0), (14, 482204.0), (15, 468571.0), (16, 458209.0), (17, 432852.0),
                 (18, 433511.0), (19, 429606.0), (20, 398854.0), (21, None), (22, None), (23, None)]
        obj = seglinreg.SegLinReg(7)
        res = obj.calculate(test1)

    def test_real2(self):
        test1 = [(0, 71904.0), (1, None), (2, 67709.0), (3, None), (4, 67285.0), (5, None), (6, 69195.0), (7, None),
                 (8, 65658.0), (9, None), (10, 69792.0), (11, None), (12, 70536.0), (13, None), (14, 70557.0),
                 (15, None), (16, 73111.0), (17, None), (18, 75862.0), (19, None), (20, 76063.0), (21, None),
                 (22, 86120.0), (23, None), (24, 93855.0), (25, None), (26, 108460.0), (27, None), (28, 125828.0),
                 (29, 138363.0), (30, 153570.0), (31, 163663.0), (32, 172087.0), (33, 193410.0), (34, 204288.0),
                 (35, 210385.0), (36, 230627.0), (37, 246666.0), (38, 261653.0), (39, 273139.0), (40, 289039.0),
                 (41, 303571.0), (42, 320915.0), (43, 335340.0), (44, 344807.0), (45, 350922.0), (46, 367048.0),
                 (47, 381013.0), (48, 387114.0), (49, None), (50, 409117.0), (51, 426089.0), (52, 438333.0), (53, None),
                 (54, 457483.0), (55, 460780.0), (56, 471158.0), (57, 472816.0), (58, 470620.0), (59, 471110.0),
                 (60, 456558.0), (61, 450758.0), (62, 449685.0), (63, 433632.0), (64, 435781.0), (65, 435683.0),
                 (66, 432804.0), (67, 434797.0), (68, 431627.0), (69, 427334.0), (70, 428529.0), (71, 420578.0),
                 (72, 442364.0), (73, 407138.0), (74, 406570.0), (75, 409014.0), (76, 414141.0), (77, 410726.0),
                 (78, 423081.0), (79, 414632.0), (80, 422843.0), (81, 426898.0), (82, 436408.0), (83, 419636.0),
                 (84, 422446.0), (85, 421977.0), (86, 427002.0), (87, 429413.0), (88, 417591.0), (89, 427550.0),
                 (90, 424619.0), (91, 430755.0), (92, 424811.0), (93, 435150.0), (94, 416189.0), (95, 423597.0),
                 (96, 413047.0), (97, 420754.0), (98, 425216.0), (99, 413913.0), (100, 425837.0), (101, 420123.0),
                 (102, 422994.0), (103, 421041.0), (104, 417075.0), (105, 430373.0), (106, 430963.0), (107, 422963.0),
                 (108, 420015.0), (109, 428893.0), (110, 422287.0), (111, 420043.0), (112, 425836.0), (113, 422760.0),
                 (114, 421627.0), (115, 419039.0), (116, 422495.0), (117, 414447.0), (118, 427455.0), (119, 420702.0),
                 (120, 431877.0), (121, 425975.0), (122, 424781.0), (123, 420973.0), (124, 418889.0), (125, 421920.0),
                 (126, 428737.0), (127, 433991.0), (128, 434087.0), (129, 417494.0), (130, 428718.0), (131, 430798.0),
                 (132, 429204.0), (133, 409045.0), (134, 428440.0), (135, 396417.0), (136, 415092.0), (137, 403576.0),
                 (138, 421255.0), (139, 400815.0), (140, 414262.0), (141, 404645.0), (142, 389748.0), (143, 419216.0),
                 (144, 411523.0), (145, 386822.0), (146, 393226.0), (147, 418498.0), (148, 393166.0), (149, 398119.0),
                 (150, 392491.0), (151, 391867.0), (152, 401225.0), (153, 391980.0), (154, 393003.0), (155, 389373.0),
                 (156, 394418.0), (157, 402042.0), (158, 408665.0), (159, 418924.0), (160, 437286.0), (161, 446457.0),
                 (162, 449352.0), (163, 472787.0), (164, 468940.0), (165, 469327.0), (166, 472071.0), (167, 468016.0),
                 (168, 467807.0), (169, 469283.0), (170, 474911.0), (171, 491348.0), (172, 510778.0), (173, 532320.0),
                 (174, 525758.0), (175, 530800.0), (176, 530458.0), (177, 533521.0), (178, 529489.0), (179, 529218.0),
                 (180, 527576.0), (181, 519932.0), (182, 504503.0), (183, 493398.0), (184, 494850.0), (185, 481679.0),
                 (186, 472237.0), (187, 471881.0), (188, 468741.0), (189, 442471.0), (190, 436289.0), (191, 418245.0),
                 (192, 397117.0), (193, 391508.0), (194, 386291.0), (195, 385817.0), (196, 381965.0), (197, 357302.0),
                 (198, 353275.0), (199, 357876.0), (200, 332120.0), (201, 326548.0), (202, 322877.0), (203, 308602.0),
                 (204, 303867.0), (205, 290147.0), (206, 286917.0), (207, 275660.0), (208, 270487.0), (209, 266210.0),
                 (210, 262437.0), (211, 264915.0), (212, 239000.0), (213, 226298.0), (214, 239081.0), (215, 224667.0),
                 (216, 223765.0), (217, 225639.0), (218, 209615.0), (219, 221696.0), (220, 216246.0), (221, 216383.0),
                 (222, 208831.0), (223, 205241.0), (224, 196052.0), (225, 193344.0), (226, 187268.0), (227, 185255.0),
                 (228, 179000.0), (229, 166115.0), (230, 166907.0), (231, 166600.0), (232, 159709.0), (233, 165959.0),
                 (234, 159896.0), (235, 149942.0), (236, 140829.0), (237, 142794.0), (238, 137999.0), (239, 135141.0),
                 (240, 131698.0), (241, 125132.0), (242, 121823.0), (243, 121424.0), (244, 118166.0), (245, 111003.0),
                 (246, 109549.0), (247, 104676.0), (248, 106413.0), (249, 101087.0), (250, 98176.0), (251, 99700.0),
                 (252, 94004.0), (253, 88756.0), (254, 88874.0), (255, 88812.0), (256, 88402.0), (257, 88914.0),
                 (258, 85681.0), (259, 86584.0), (260, 85051.0), (261, 80567.0), (262, 80339.0), (263, 81354.0),
                 (264, 77276.0), (265, 80721.0), (266, 82002.0), (267, 75909.0), (268, 74948.0), (269, 80033.0),
                 (270, 73773.0), (271, 75631.0), (272, 73823.0), (273, 76962.0), (274, 75843.0), (275, 76436.0),
                 (276, 75536.0), (277, 75628.0), (278, 75791.0), (279, 73271.0), (280, 73552.0), (281, 71447.0),
                 (282, 72385.0), (283, 70651.0), (284, 73755.0), (285, 73420.0), (286, 72158.0), (287, 71004.0)]
        obj = seglinreg.SegLinReg(3)
        res = obj.calculate(test1)

    def test_real3(self):
        test = [(0, 69510.4507689), (1, None), (2, 69505.6840223), (3, None), (4, 68706.4540974), (5, None),
                (6, 71148.7973211), (7, None), (8, 68622.8445775), (9, None), (10, 66550.1819914), (11, None),
                (12, 71132.4440549), (13, None), (14, 73209.6825078), (15, None), (16, 79484.480693), (17, None),
                (18, 81533.5641806), (19, None), (20, 76469.6592558), (21, None), (22, 80414.1142629), (23, None),
                (24, 91121.5636476), (25, None), (26, 103109.296503), (27, None), (28, 124882.940696),
                (29, 133120.59759), (30, 134572.214626), (31, 147270.321862), (32, 156656.326482), (33, 178763.655991),
                (34, 186984.891221), (35, 195573.464168), (36, 233798.167846), (37, 229480.018082), (38, 236193.568477),
                (39, 258909.653055), (40, 275718.700339), (41, 279651.409514), (42, 291099.532944), (43, 297906.780718),
                (44, 312325.236646), (45, 322399.562433), (46, 337573.078346), (47, 353235.865614), (48, 360402.791337),
                (49, None), (50, 387055.55907), (51, 409350.587975), (52, 406974.672872), (53, None),
                (54, 435672.479923), (55, 427094.892147), (56, 435872.011227), (57, 438772.217883), (58, 434708.434042),
                (59, 428541.949537), (60, 404147.774514), (61, 392572.905925), (62, 391886.706002), (63, 399068.951269),
                (64, 376154.995021), (65, 381965.065873), (66, 381425.31456), (67, 369282.690686), (68, 368572.998739),
                (69, 376372.044439), (70, 375404.01074), (71, 383302.01975), (72, 368977.775774), (73, 365020.610099),
                (74, 372243.090093), (75, 361020.612113), (76, 359566.722852), (77, 376839.13545), (78, 367281.547363),
                (79, 369924.618023), (80, 364777.199298), (81, 372070.218685), (82, 359200.932362), (83, 369708.933349),
                (84, 369771.971992), (85, 368152.396362), (86, 357306.764141), (87, 356295.175993), (88, 366900.098638),
                (89, 375811.361924), (90, 361108.388692), (91, 363964.372442), (92, 363579.380742), (93, 370501.297063),
                (94, 356247.633601), (95, 353175.912417), (96, 349122.686675), (97, 354614.033404), (98, 357064.190523),
                (99, 355003.608326), (100, 363424.589861), (101, 361050.730829), (102, 362806.736792),
                (103, 363466.551082), (104, 366378.684271), (105, 362872.02329), (106, 373629.039449),
                (107, 367691.261455), (108, 360574.302379), (109, 363455.431211), (110, 365117.338511),
                (111, 368327.685666), (112, 364502.324634), (113, 358866.39384), (114, 373312.684432),
                (115, 365554.365862), (116, 367854.04538), (117, 372039.331199), (118, 354210.748284),
                (119, 358571.72956), (120, 364976.330877), (121, 347206.540892), (122, 367951.020997),
                (123, 367100.673093), (124, 357683.382229), (125, 367402.617478), (126, 358974.047388),
                (127, 380166.516089), (128, 371386.510196), (129, 363440.89795), (130, 362856.958518),
                (131, 353042.458582), (132, 346555.645926), (133, 347451.600602), (134, 358266.314548),
                (135, 360459.783478), (136, 361214.196787), (137, 354399.557726), (138, 357509.362841),
                (139, 360690.588785), (140, 372923.796594), (141, 364381.108079), (142, 360213.525323),
                (143, 371514.259732), (144, 361820.418824), (145, 360163.41624), (146, 347328.699478),
                (147, 362489.055071), (148, 350599.961998), (149, 353560.850894), (150, 348731.887046),
                (151, 351110.000152), (152, 353431.867367), (153, 345008.190382), (154, 339754.891279),
                (155, 331720.787093), (156, 326241.61505), (157, 324522.83627), (158, 328477.450384),
                (159, 353293.908162), (160, 358052.682748), (161, 359426.917128), (162, 370906.710648),
                (163, 369569.50593), (164, 370620.50616), (165, 370563.934889), (166, 382315.731283),
                (167, 360592.398131), (168, 355762.190439), (169, 365124.491368), (170, 378756.971381),
                (171, 389736.464576), (172, 418168.970632), (173, 431843.054181), (174, 430795.184084),
                (175, 435291.448704), (176, 423033.843658), (177, 432859.499015), (178, 423398.970154),
                (179, 426803.04412), (180, 420960.114392), (181, 419236.183346), (182, 421040.99071),
                (183, 417630.797843), (184, 394013.374023), (185, 404158.895341), (186, 412994.967477),
                (187, 394376.423473), (188, 388026.247226), (189, 387723.823897), (190, 360114.511816),
                (191, 358262.801208), (192, 350013.341038), (193, 338011.674437), (194, 333484.329805),
                (195, 319974.102986), (196, 317237.070556), (197, 301491.133481), (198, 299663.36578),
                (199, 293453.508487), (200, 270478.999956), (201, 256016.489509), (202, 262751.424419),
                (203, 253254.636123), (204, 254397.030394), (205, 252277.280892), (206, 249553.954565),
                (207, 225563.41016), (208, 237791.539908), (209, 230655.023311), (210, 224526.571034),
                (211, 213312.566406), (212, 210439.172792), (213, 198885.018495), (214, 188376.603376),
                (215, 198379.048764), (216, 190632.453617), (217, 196109.320469), (218, 186779.209451),
                (219, 188081.735425), (220, 191638.331918), (221, 185832.6326), (222, 174874.145909),
                (223, 171456.752941), (224, 165192.245672), (225, 177965.074864), (226, 159765.813437),
                (227, 151891.654819), (228, 156638.304349), (229, 153741.020874), (230, 154540.353209),
                (231, 146923.498373), (232, 135893.204821), (233, 127548.312046), (234, 137586.670885),
                (235, 124906.526232), (236, 124602.616568), (237, 130438.797418), (238, 122503.510657),
                (239, 128125.106931), (240, 122894.32492), (241, 114950.809604), (242, 112370.932257),
                (243, 106676.98456), (244, 108991.263975), (245, 116860.635346), (246, 106657.801011),
                (247, 115294.612254), (248, 102678.989704), (249, 92136.282744), (250, 94099.2745416),
                (251, 95764.9134576), (252, 88805.5551867), (253, 94259.7802251), (254, 98946.1559517),
                (255, 89740.9688949), (256, 86247.550213), (257, 88800.1877013), (258, 86506.7567866),
                (259, 94904.4827482), (260, 81336.4190777), (261, 77492.5331845), (262, 80768.2635672),
                (263, 91560.677563), (264, 89493.5157987), (265, 79818.3829299), (266, 76920.3791612),
                (267, 79338.7721022), (268, 80797.1735159), (269, 85740.7223989), (270, 80874.9965039),
                (271, 88904.7061624), (272, 77645.463543), (273, 94586.5444507), (274, 79622.8861028),
                (275, 76753.7953382), (276, 77867.0637845), (277, 83562.8407968), (278, 79836.2892133),
                (279, 77511.4213621), (280, 79373.058903), (281, 72470.3864589), (282, 71632.6321596),
                (283, 69155.6725561), (284, 70844.7810467), (285, 75333.790165), (286, 75915.9181334),
                (287, 77417.6798966)]
        obj = seglinreg.SegLinReg(7)
        res = obj.calculate(test)


if __name__ == '__main__':
    unittest.main()


