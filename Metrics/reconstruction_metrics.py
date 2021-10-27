# reconstruction metrics
"""
uqi,mse,rmse,psnr,ergas,scc,sam,msssim
"""
from sewar.full_ref import uqi, mse, rmse, psnr, ergas, scc, sam, msssim


class IQM():
    def __init__(self, real_image, recon_image, channels=1):
        self.real = real_image
        self.recon = recon_image
        self.channel = channels

    def scores(self):
        MSE = []
        RMSE = []
        PSNR = []
        ERGAS = []
        SCC = []
        SAM = []
        UQI = []
        SSIM = []

        for i in range(self.real.shape[0]):
            real = self.real[i]
            recon = self.recon[i]
            MSE.append(mse(real, recon))
            RMSE.append(rmse(real, recon))
            PSNR.append(psnr(real, recon, MAX=max(recon)))
            ERGAS.append(ergas(real, recon, r=4, ws=8))
            SCC.append(scc(real, recon, ws=8))
            SAM.append(sam(real, recon))
            UQI.append(uqi(real, recon, ws=8))
            # VIFP.append(vifp(real, recon, sigma_nsq=2))

            # if self.channel == 1:
            #     SSIM.append(ssim(real, recon, ws=8, K1=0.01,
            #                      K2=0.03, MAX=max(recon)))

            if self.channel > 1:
                SSIM.append(msssim(real, recon, ws=8,
                                   K1=0.01, K2=0.03, MAX=max(recon)))

        data = {"i": i, "MSE": MSE, "RMSE": RMSE, "PSNR": PSNR, "SSIM": SSIM,"ERGAS": ERGAS, "SCC": SCC, "SAM": SAM, "UQI": UQI}

        return data
