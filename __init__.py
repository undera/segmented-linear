from graphite.render.functions import SeriesFunctions
import seglinreg

SeriesFunctions['segLinReg'] = seglinreg.seg_lin_reg
SeriesFunctions['segLinRegAuto'] = seglinreg.seg_lin_reg_auto

