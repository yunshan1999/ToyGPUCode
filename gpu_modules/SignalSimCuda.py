# #############################################################
# Signal simulation details shall be written here!
# this currently is just a test and demonstration!
# #############################################################
# Created @ 2021-04-05
# by Yunshan Cheng
# email: yunshancheng1@gmail.com
###############################################################

pandax4t_signal_sim = """
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
extern "C" {
__device__ float gpu_truncated_gaussian(curandState_t *rand_state, float mean, float sigma, float lower, float upper)
{
    float x = lower;
    while(x>=upper||x<=lower)
    {
        x = curand_normal(rand_state)*sigma + mean;
    }
    return x;
}

__device__ int gpu_binomial(curandState_t *rand_state, int num_trials, float prob_success)
{
    int x = 0;
    if(prob_success<=0) return 0;
    if(prob_success>=1) return num_trials;
    // do a gaussian approximation if:
    // np(1-p)>10
    if ( ((float)num_trials)*prob_success*(1.-prob_success)>10 )
    {
        x = (int) (curand_normal(rand_state) * sqrtf( ((float)num_trials)*prob_success*(1.-prob_success) ) + ((float)num_trials)*prob_success);
    }
    else
    {
        for(int i = 0; i < num_trials; i++) 
        {
            if(curand_uniform(rand_state) < prob_success)
                x += 1;
        }
    }
    return x;
}

__device__ float interpolate1d(float x, float * array_x, float * array_y){
    int ndimx = (int)array_x[0];
    int ndimy = (int)array_y[0];
    if(ndimx != ndimy){
        printf("two arrays not in same dimension!");
        return 0.;
    }
    else{
        if(x < array_x[1])return array_y[1];
        else if(x>array_x[ndimx])return array_y[ndimx];
        else{
            int i;
            for(i = 1; i <= ndimx ; i++){
            if(x > array_x[i]);
            else break;
            }
            float xmin = array_x[i-1];
            float xmax = array_x[i];
            float ymin = array_y[i-1];
            float ymax = array_y[i];
            float temp = ymin + (ymax - ymin)/(xmax - xmin) * (x - xmin);
            return temp;
        }

    }
}

__device__ float get_tritium_energy_weight(float energy, float * nuisance_par){

    float flat = nuisance_par[4]; 
    float T = energy; 
    float Q = 18.5906, pi = 3.1415926; 
    if (T > Q||T < 0)return flat; 
    float m = 0.511e3; 
    float P = sqrt(2 * m * T);  
    float E = T + m; 
    float eta = 2.*1./137*E/P; 
    float F = 2*pi*eta/(1-exp(-2*pi*eta)); 
    return double(F * P * E * (Q - T) * (Q - T) * 0.00001 + flat);
}

__device__ float get_fano_factor(bool simuTypeNR, float density, int Nq_mean, float E_drift){
    if(simuTypeNR){
        return 1.;
    }
    else{
        float fano = 0.12707 - 0.029623 * density - 0.0057042 * powf(density, 2.)
        + 0.0015957 * powf(density, 3.);
        fano += 0.0015 * sqrtf(Nq_mean) * powf(E_drift, 0.5);
        return fano;
    }
}

__device__ void get_yield_pars(bool simuTypeNR, float E_drift, float energy, float density, float * pars, float * free_pars){
    //This function calculates Lindhaed, W, Nex/Ni, recomb fraction mean and delta and returns them all 
    //ATTENTION: Just to clarify.NuisParam&FreeParam are NESTv2 Pars, not the ones we are going to fit ni this model. Here we just regard them as intrinsic fixed pars.

    float nest_avo = 6.0221409e+23;
    float molar_mass = 131.293;
    float atom_num = 54.;
    float eDensity = (density/molar_mass) * nest_avo * atom_num; 
    float Wq_eV = 18.7263 - 1.01e-23 * eDensity;
    Wq_eV *= 1.1716263232;
    
    if(simuTypeNR){
        float NuisParam[11] = {11.,1.1,0.0480,-0.0533,12.6,0.3,2.,0.3,2.,0.5,1.};
        float Nq = NuisParam[0] * powf(energy, NuisParam[1]);
        float ThomasImel =
            NuisParam[2] * powf(E_drift, NuisParam[3]) * powf(density / 2.90, 0.3);
        float Qy = 1. / (ThomasImel*powf(energy+NuisParam[4],NuisParam[9]));
        Qy *= 1. - 1. / powf(1. + powf((energy / NuisParam[5]), NuisParam[6]),NuisParam[10]);
        float Ly = Nq / energy - Qy;
        if (Qy < 0.0) Qy = 0.0;
        if (Ly < 0.0) Ly = 0.0;
        float Ne = Qy * energy;
        float Nph = Ly * energy *
              (1. - 1. / (1. + powf((energy / NuisParam[7]), NuisParam[8])));
        Nq = Nph + Ne;
        float Ni = (4. / ThomasImel) * (expf(Ne * ThomasImel / 4.) - 1.);
        float Nex = (-1. / ThomasImel) * (4. * expf(Ne * ThomasImel / 4.) -
                                           (Ne + Nph) * ThomasImel - 4.);
        float elecFrac = Ne/(Ne+Nph);
        float NexONi = Nex / Ni;
        float rmean = 1. - Ne / Ni;
        float FreeParam[] = {1.,1.,0.1,0.5,0.19,2.25};
        float omega = FreeParam[2] * expf(-0.5 * powf(elecFrac - FreeParam[3],2.)/(FreeParam[4] * FreeParam[4]));
        if(omega < 0.)omega = 0.;
        double L = (Nq / energy) * Wq_eV * 1e-3;
        pars[0] = Wq_eV;
        pars[1] = L;
        pars[2] = NexONi;
        pars[3] = rmean + free_pars[6] + free_pars[7]*energy + free_pars[8]*energy*energy;
        pars[4] = omega * free_pars[5];
        if(pars[3]<0.)pars[3] = 0.;
        if(pars[3]>1.)pars[3] = 1.;
        if(pars[4]<0.)pars[4] = 0.;
        return;
    }
    else{
        // float Wq_eV =
      1.9896 + (20.8 - 1.9896) / (1. + powf(density / 4.0434, 1.4407));

        float QyLvllowE = 1e3 / Wq_eV + 6.5 * (1. - 1. / (1. + powf(E_drift / 47.408, 1.9851)));
        float HiFieldQy =
          1. + 0.4607 / powf(1. + powf(E_drift / 621.74, -2.2717), 53.502);
        float QyLvlmedE =
          32.988 -
          32.988 /
              (1. + powf(E_drift / (0.026715 * expf(density / 0.33926)), 0.6705));
      QyLvlmedE *= HiFieldQy;
        float DokeBirks =
          1652.264 +
          (1.415935e10 - 1652.264) / (1. + powf(E_drift / 0.02673144, 1.564691));
        float Nq = energy * 1e3 /
                  Wq_eV;  //( Wq_eV+(12.578-Wq_eV)/(1.+powf(energy/1.6,3.5)) );
        float LET_power = -2.;
        float QyLvlhighE = 28.;
      //      if (density > 3.) QyLvlhighE = 49.; Solid Xe effect from Yoo. But,
      //      beware of enabling this line: enriched liquid Xe for neutrinoless
      //      double beta decay has density higher than 3g/cc;
        float Qy = QyLvlmedE +
                  (QyLvllowE - QyLvlmedE) /
                      powf(1. + 1.304 * powf(energy, 2.1393), 0.35535) +
                  QyLvlhighE / (1. + DokeBirks * powf(energy, LET_power));
        if (Qy > QyLvllowE && energy > 1. && E_drift > 1e4) Qy = QyLvllowE;
        float Ly = Nq / energy - Qy;
        float Ne = Qy * energy;
        float Nph = Ly * energy;
        float alpha = 0.067366 + density * 0.039693;
        float NexONi = alpha * erff(0.05 * energy);
        float Nex = Nq * (NexONi) / (1. + NexONi);
        float Ni = Nq * 1. / (1. + NexONi);
        float rmean = 1 - Ne / Ni;
        float elecFrac = Ne/(Ne+Nph);

        float ampl = 0.14 + (0.043 - 0.14)/(1. + powf(E_drift/1210.,1.25)); //0.086036+(0.0553-0.086036)/powf(1.+powf(E_drift/295.2,251.6),0.0069114); //pair with GregR mean yields model
        if(ampl < 0.)ampl = 0.;
        float wide = 0.205; //or: FreeParam #2, like amplitude (#1)
        float cntr = 0.5; //0.41-45 agrees better with Dahl thesis. Odd! Reduces fluctuations for high e-Frac (high EF,low E). Also works with GregR LUX Run04 model. FreeParam #3
        //for gamma-rays larger than 100 keV at least in XENON10 use 0.43 as the best fit. 0.62-0.37 for LUX Run03
        float skew = -0.2; //FreeParam #4
        float mode = cntr + sqrtf(2./M_PI)*skew*wide/sqrtf(1.+skew*skew);
        float norm = 1./(expf(-0.5*powf(mode-cntr,2.)/(wide*wide))*(1.+erff(skew*(mode-cntr)/(wide*sqrtf(2.))))); //makes sure omega never exceeds ampl
        float omega = norm*ampl*expf(-0.5*powf(elecFrac-cntr,2.)/(wide*wide))*(1.+erff(skew*(elecFrac-cntr)/(wide*sqrtf(2.))));
        if(omega<0.)omega = 0;
        pars[0] = Wq_eV;
        pars[1] = 1.;
        pars[2] = NexONi;
        pars[3] = rmean + free_pars[6] + free_pars[7]*energy + free_pars[8]*energy*energy;
        pars[4] = omega * free_pars[5];
        if(pars[3]<0.)pars[3] = 0.;
        if(pars[3]>1.)pars[3] = 1.;
        if(pars[4]<0.)pars[4] = 0.;
        return;
    }
}

__device__ float get_elife(curandState_t *rand_state, int typeFlag){
    //return 600.;
    
    //float * elife;
    //float * duration;
    //if(typeFlag==0)//tritium
    //{
        float elife[] = 
        {140, 883.37, 871.64, 870.78, 919.58, 948.48, 953.05, 1043.06, 1094.79, 1139.86, 1050.64, 1060.68, 1058.91, 1047.46, 1134.27, 1127.89, 1113.11, 1157.41, 1156.50, 1193.77, 1233.66, 1095.81, 1444.49, 1052.65, 951.05, 936.74, 940.68, 975.83, 1040.61, 1048.84, 1101.86, 1057.57, 1130.22, 1107.60, 1181.09, 1188.38, 1214.67, 1234.44, 1244.70, 1255.44, 1266.57, 1276.68, 1287.44, 1298.20, 1308.85, 1319.29, 1330.05, 1337.53, 1340.66, 1343.93, 1351.29, 1362.05, 1370.19, 1374.29, 1377.10, 1382.16, 1386.45, 1387.78, 1392.37, 1400.68, 1411.42, 1418.64, 1423.91, 1431.14, 1436.46, 1443.29, 1450.92, 1456.50, 1461.41, 1466.73, 1472.22, 1475.48, 1479.41, 1484.07, 1489.48, 1496.16, 1506.31, 1514.05, 1238.99, 1244.83, 1249.04, 1258.90, 1262.62, 1267.83, 1275.46, 1283.21, 1291.05, 1298.64, 1302.84, 1304.79, 1308.30, 1311.77, 1315.27, 1321.79, 1329.52, 1337.24, 1344.59, 1348.40, 1349.50, 1353.38, 1360.41, 1367.31, 1375.93, 1382.41, 1386.29, 1390.81, 1394.71, 1395.40, 1397.86, 1403.87, 1407.70, 1414.47, 1418.46, 1420.14, 1421.20, 1421.64, 1422.30, 1422.77, 1424.52, 1429.92, 1435.83, 1439.25, 1441.05, 1443.74, 1447.64, 1452.19, 1456.01, 1460.80, 1467.36, 1471.17, 1476.25, 1495.13, 1513.88, 1559.41, 1588.78, 1617.81, 1646.72, 1675.33, 1704.26, 1733.23, 1700.26};
        float duration[] = 
        {140, 718.18, 1337.27, 759.43, 646.04, 1472.82, 1401.57, 702.33, 1003.81, 471.47, 1184.33, 1395.77, 1266.68, 611.05, 640.38, 1428.67, 1442.69, 1308.33, 1197.74, 1323.49, 1441.18, 345.28, 238.52, 2808.99, 1220.11, 1218.40, 2721.25, 1390.12, 1432.68, 1427.15, 1442.54, 1271.54, 1455.89, 1342.15, 1046.69, 1507.85, 1431.71, 1306.28, 1454.58, 1436.50, 1301.47, 1428.10, 1474.11, 1433.91, 1376.31, 1431.53, 1461.34, 499.32, 235.39, 553.96, 1434.38, 1409.21, 781.06, 292.52, 260.45, 977.45, 144.38, 185.49, 805.10, 1438.20, 1462.90, 485.39, 898.77, 1026.44, 404.18, 1437.41, 620.12, 755.47, 570.27, 865.40, 615.54, 257.37, 307.34, 920.18, 503.29, 1300.60, 1439.32, 649.39, 651.58, 1349.39, 214.30, 840.65, 538.96, 1402.36, 1440.39, 1437.51, 1398.31, 1430.93, 127.01, 483.65, 755.83, 533.34, 269.55, 1445.00, 1437.48, 1433.13, 1225.29, 161.51, 244.61, 1180.32, 1438.81, 1130.66, 1411.38, 1002.71, 428.91, 1245.53, 192.66, 55.50, 328.99, 312.30, 1082.30, 1439.39, 43.20, 206.14, 58.35, 85.95, 124.07, 40.31, 572.99, 1436.41, 760.61, 436.64, 174.34, 828.49, 588.87,1041.05, 342.91, 1439.92, 908.77, 459.46, 1435.10, 1414.41, 1440.70, 1438.75, 1412.41, 1471.12, 1400.88, 1439.13, 1434.57, 1441.65, 983.21};

    //}
    //else if(typeFlag==1)//AmBe
    //{
      //  elife = {3,672.9,658.4,640.0};
      //  duration = {3,41.371,24.7483,33.3846};
    //}
    //else//DD
    //{
      //  elife = {1,683.896};
      //  duration = {41.6};
    //}
    float temp = 100.;
    int n = (int)elife[0];
    float duration_tot = 0.;
    for(int i = 1; i <= n ; i++){
        duration_tot +=duration[i];
    }
    for(int i = 1; i <= n ; i++){
        duration[i] /= duration_tot;
    }
    float dice = curand_uniform(rand_state);
    float percCount = 0.;
    for(int i = 1; i <= n ; i++){
        percCount += duration[i];
       if(dice<=percCount){
            temp = elife[i];
            //if(temp<=0.)printf("%d, %f\\n",i,temp);
            return temp;
            break;
        }
    }
}
        
__device__ float get_g1_true(float dt, float g1mean){
    float A = 5.64008e2;
    float B = 3.92408e-1;
    float C = -1.65968e-4;
    float D = 6.69171e-7;
    float qS1maxmean = 803.319;
    float qS1max = A + B * dt + C * dt * dt + D * dt * dt * dt;
    float g1_true = g1mean * qS1max/qS1maxmean;
    return g1_true;
}

__device__ float get_s1_efficiency(int nHitsS1){
    float hit[] = {20, 
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    float eff[] = {20,
                    0.,0.,0.3442748091603054,0.6625954198473284,0.7839694656488551,
                    0.836641221374046,0.8412213740458017,0.8893129770992368,
                    0.9030534351145039,0.9053435114503818,0.919083969465649,
                    0.9328244274809161,0.9442748091603055,0.9511450381679392,
                    0.9534351145038169,0.9534351145038169,0.9603053435114505,
                    0.9648854961832063,0.9717557251908397,0.9763358778625955
                    };
    float Eff = interpolate1d((float)nHitsS1, hit, eff);
    return Eff;
}

__device__ void get_bls(float pulseAreaS1, float * bias, float * fluctuation){
    float phe[] = {
        31, 
        1, 2, 3, 4, 5, 7, 10 ,15 ,20 ,
        25. ,30. ,40. ,50. ,60. ,70. ,80. ,
        90. ,100.,120.,150.,175.,200.,250.,
        300.,350.,400.,500.,600.,700.,800.,900.
    };
    float Bias[] = {
        31,
        0.86113054 , 0.88597399 , 0.89066607 , 0.89654213 , 0.89583844 ,
        0.89926350 , 0.90153772 , 0.90349376 , 0.90529740 , 0.90669477 , 
        0.90670788 , 0.90906495 , 0.91094166 , 0.91262519 , 0.91373128 , 
        0.91592121 , 0.91728061 , 0.91894722, 0.92216468, 0.92646188, 
        0.92915821, 0.93229359, 0.93787754, 0.94296849, 0.94764942, 0.95168173, 
        0.95867330, 0.96446264, 0.96942329, 0.97360170, 0.97696060
    };
    float Fluctuation[] = {
        31,
        0.34581023,0.21751384,0.17195459,0.14183772,0.12760288,
        0.10751605,0.088833449,0.070189188,0.060589716,0.053495033,
        0.049544983,0.042539377,0.037536195,0.033775607,0.031350559,
        0.028949335,0.027120965,0.025658420,0.022583934,0.019707454,
        0.017826143,0.016797188,0.014447413,0.012827946,0.011236941,
        0.010328728,0.0086439596,0.0075480391,0.0067549600,0.0060658234,0.0055888618    
    };
    
    bias[0] = interpolate1d(pulseAreaS1, phe, Bias);
    fluctuation[0] = interpolate1d(pulseAreaS1, phe, Fluctuation);
}

__device__ float get_g1_inverse_correction_factor(float dt){
    float A = 5.64008e2;
    float B = 3.92408e-1;
    float C = -1.65968e-4;
    float D = 6.69171e-7;
    float qS1maxmean = 803.319;
    float qS1max = A + B * dt + C * dt * dt + D * dt * dt * dt;
    return qS1maxmean/qS1max;
}

__device__ float get_g2_inverse_correction_factor(float dt, float eLife_us){
    return expf(dt/eLife_us);
}

__device__ float get_s1_tagging_eff(bool simuTypeNR, float x){
    if(simuTypeNR)return 1.;
    else
    {
        float mu = -3.58173239;
        float sigma = 2.92726842;
        return 1 / (expf(-(x-mu)/sigma) + 1);
    }
}

__device__ float get_cs1_qualitycut_eff(float x){
    float mu = -1.25143721;
    float sigma = 4.22550541;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__device__ float get_cs2b_qualitycut_eff(float x){
    float mu = 162.46632354;
    float sigma = 93.36755591;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__global__ void signal_simulation(
        int *seed,
        float *input,
        float *nuisance_par,
        float *output)
{
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s;
    curand_init((*seed)*iteration,2, 0, &s);
    
    // get the number of trials
    int num_trials      = (int) input[0];
    
    //printf("number of trials = %d", num_trials);
    
    // keep only the GPU nodes that satisfy the number of trial
    if (iteration>num_trials) return;
       
    // print
    //printf("This is the %d-th GPU thread!\\n", iteration);
    //return;

    //to determine whether it is ER or NR
    bool simuTypeNR = false;
    int typeFlag = 0;

    //get energy randomly
    float lower = 0.1;
    float upper = 70.;
    
    float energy = curand_uniform(&s)*(upper-lower)+lower;
    float weight = get_tritium_energy_weight(energy, &nuisance_par[0]);
    if(weight<=0.)weight = 0.;
    
    //get detector parameters

    float g1 = nuisance_par[0]; //hit per photon
    float sPEres = 0.3; //pmt single phe resolution (gaussian assumed)
    float P_dphe = 0.2; //probab that 1 phd makes 2nd phe 
    float SEG = nuisance_par[2]; //single electron gain, num of photon/electron 
                                        //before elife decay&EEE
    float ExtraEff = nuisance_par[1]; //electron extraction eff
    float deltaG = 7.; //SEG resolution
    float eLife_us = get_elife(&s,typeFlag); //the drift electron in microsecond
    
    //float hit_thr = 1.;
    float s1_thr = 2.;
    //float s2_thr_beforecorrection = 80;
    //float s2_thr = 80; //s2 threshold in phe
    float s1_max = 120;
    float s2_max = 20000;
    float E_drift = 115.6; //drift electric field in V/cm
    float driftvelocity = 1.2; //in mm/microsecond
    //float dt_min = 20;//in ms
    //float dt_max = 770;//in ms
    float zup = 1200. - 20. * driftvelocity;
    float zdown = 1200. - 770. * driftvelocity;
    float density = 2.8611; //density of liquid xenon
    float TopDrift = 1200.; // mm 
    float bottom_fraction = nuisance_par[3];

    float pars[5] = {0.};//={W_eV,Nex/Ni, rmean, rdelta}
    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0], &nuisance_par[0]);
    float w_eV = pars[0];
    float L = pars[1];
    float NexONi = pars[2];
    float rmean = pars[3];
    float deltaR = pars[4];

    //let the simulation begin

    // 1) get mean quanta number
    int Nq_mean = (int)(energy / w_eV * 1000.);
    
    // 2) get actual quanta number by fluctuating quanta number with fano factor
    float Fano = get_fano_factor(simuTypeNR, density, Nq_mean, E_drift);
    int Nq_actual = (int)(curand_normal(&s)*sqrtf(Fano * Nq_mean) + Nq_mean);
    if(Nq_actual <= 0. )Nq_actual = 0.;
    
    // 3) get quanta number after lindhard factor fluctuation
    int Nq = gpu_binomial(&s, Nq_actual, L);
    if(Nq <= 0.)Nq = 0.;
    
    // 4）get exciton ratio and do fluctuation
    int Ni = gpu_binomial(&s, Nq, 1/(1 +NexONi));
    int Nex = Nq - Ni;

    // 5) get recomb fraction fluctuation
    float r = curand_normal(&s)*deltaR + rmean;
    if(r >= 1. )r = 1.;
    else if(r <= 0.)r = 0.;

    // 6) get photon and electron numbers
    int Ne = gpu_binomial(&s, Ni, 1 - r);
    int Nph = Ni + Nex - Ne;

    // 7) get drift time 
    float truthz = zdown + curand_uniform(&s) * (zup - zdown);
    float dt = (TopDrift - truthz) / driftvelocity;

    // 8) get g1 hit(phd) number
    float g1_true = get_g1_true(dt,g1);
    int nHitsS1 = gpu_binomial(&s, Nph, g1_true);
    //if(nHitsS1 < hit_thr)nHitsS1 = 0.;
    //if(nHitsS1 < hit_thr)return;
    //if(nHitsS1 <= 0.)nHitsS1 = 0.;
    
    // 9) get s1 in phe #, consider float phe
    int NpheS1 = nHitsS1 + gpu_binomial(&s, nHitsS1, P_dphe);
    
    // 10) s1 pulse area, with pmt resolution
    float pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    //if(pulseAreaS1 <= 0.)return;
    if(pulseAreaS1 <= 0.)pulseAreaS1 = 0.;

    // 11) biased s1 pulse area (same in s2 bias)
    float biasS1, fluctuationS1;
    get_bls(pulseAreaS1, &biasS1, &fluctuationS1);
    float pulseAreaBiasS1 = pulseAreaS1 * (curand_normal(&s)*fluctuationS1 + biasS1);
    //if(pulseAreaBiasS1 <= 0.)return;
    if(pulseAreaBiasS1 <= 0.)pulseAreaBiasS1 = 0.;

    // 12) corrected s1 pulse area
    float InversedS1Correction = get_g1_inverse_correction_factor(dt);
    float pulseAreaS1Cor = pulseAreaBiasS1 * InversedS1Correction;
    //if(pulseAreaS1Cor <= 0.)pulseAreaS1Cor = 0.;
    //if(pulseAreaS1Cor <= s1_thr)return;
    if(pulseAreaS1Cor <= s1_thr || pulseAreaS1Cor > s1_max)return;

    // 13) do electron drifting and extraction
    int Nee = gpu_binomial(&s, Ne, expf(-dt / eLife_us) * ExtraEff);

    // 14) get s2 hit number 
    int nHitsS2 = (int)(curand_normal(&s)*sqrtf((float)Nee)*deltaG + SEG*(float)Nee);
    //if(nHitsS2 <= 0.)return;
    if(nHitsS2 <= 0.)nHitsS2 = 0.;

    // 15) get s2 phe
    int NpheS2 = nHitsS2 + gpu_binomial(&s, nHitsS2, P_dphe);
    
    // 16) s2 pulse area, with pmt resolution
    float pulseAreaS2 = curand_normal(&s)*sqrtf(NpheS2)*sPEres + NpheS2; 
    if(pulseAreaS2 <= 0.)pulseAreaS2 = 0.;

    // 17) biased s2 pulse area 
    float biasS2, fluctuationS2;
    get_bls(pulseAreaS2, &biasS2, &fluctuationS2);
    float pulseAreaBiasS2 = pulseAreaS2 * (curand_normal(&s)*fluctuationS2 + biasS2);
   // if(pulseAreaBiasS2 <=0.)return;
    //if(pulseAreaBiasS2 <= s2_thr_beforecorrection)return;

    // 18）corrected s2 pulse area
    float InversedS2Correction = get_g2_inverse_correction_factor(dt, eLife_us);
    float pulseAreaS2Cor = pulseAreaBiasS2 * InversedS2Correction;    
    //if(pulseAreaS2Cor <= s2_thr)return;
    if(pulseAreaS2Cor > s2_max)return;
    float pulseAreaS2CorB = pulseAreaS2Cor * bottom_fraction;

    weight *= get_s1_tagging_eff(simuTypeNR, pulseAreaS1Cor);
    weight *= get_cs1_qualitycut_eff(pulseAreaS1Cor); 
    weight *= get_cs2b_qualitycut_eff(pulseAreaS2CorB);

    int histFlag = (int)*(input+1);
    if(histFlag == 0)
    {
        int xBinning = (int)*(input+2);
        float xMin = (float)*(input+3);
        float xMax = (float)*(input+4);
        float xStep = (xMax - xMin)/(float)xBinning;
        int yBinning = (int)*(input+5);
        float yMin = (float)*(input+6);
        float yMax = (float)*(input+7);
        float yStep = (yMax - yMin)/(float)yBinning;

        atomicAdd(&output[0], 1.);
        //get values, overflows and underflows in output[1]-output[9]
        float xvalue = (float)pulseAreaS1Cor;
        float yvalue = (float)log10f(pulseAreaS2CorB);
    
        if(xvalue<xMin && yvalue>=yMax)atomicAdd(&output[2], 1.);
        else if(xvalue>xMin && xvalue<xMax && yvalue>=yMax)atomicAdd(&output[3], 1.);
        else if(xvalue>=xMax && yvalue>=yMax)atomicAdd(&output[4], 1.);
        else if(xvalue<xMin && yvalue>yMin && yvalue<yMax)atomicAdd(&output[5], 1.);
        else if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
        {
            int xbin = (int) ((xvalue - xMin)/xStep) + 1; 
            int ybin = (int) ((yvalue - yMin)/yStep) + 1;
            //weight *= get_s1_efficiency(nHitsS1);
            //if(weight<0.){weight = 0.;}
            atomicAdd(&output[6], 1.);
            atomicAdd(&output[1], weight);
            int index = 10+(ybin-1)*xBinning+xbin;
            int errindex = 10+xBinning*yBinning+((ybin-1)*xBinning+xbin);
            atomicAdd(&output[index], weight);
            atomicAdd(&output[errindex],weight*weight);
        }
        else if(xvalue>=xMax && yvalue>yMin && yvalue<yMax)atomicAdd(&output[7], 1.);
        else if(xvalue<xMin && yvalue<yMin)atomicAdd(&output[8], 1.);
        else if(xvalue>xMin && xvalue<xMax && yvalue<yMin)atomicAdd(&output[9], 1.);
        else if(xvalue>=xMax && yvalue<yMin)atomicAdd(&output[10], 1.);
    }
    else if(histFlag == 1)
    {
    output[num_trials*0+iteration] = energy;
    output[num_trials*1+iteration] = Nph;
    output[num_trials*2+iteration] = Ne;
    output[num_trials*3+iteration] = pulseAreaS1Cor;
    output[num_trials*4+iteration] = pulseAreaS2CorB;
    output[num_trials*5+iteration] = weight;
    }
    
}
}

"""
