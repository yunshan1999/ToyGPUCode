# #############################################################
# Signal simulation details shall be written here!
# this currently is just a test and demonstration!
# #############################################################
# Created @ 2021-04-05
# by Yunshan Cheng
# email: yunshancheng1@gmail.com
###############################################################

pandax4t_signal_sim_dd = """
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
extern "C" {
__device__ float gpu_truncated_gaussian(curandState_t *rand_state, float mean, float sigma, float lower, float upper)
{
    float x = mean;
    for(int i=0;i<1e2;i++)
    {
        x = curand_normal(rand_state)*sigma + mean;
        if (x<upper && x>lower) break;
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

__device__ float interpolate1d_bin(float x, float * array_x, int ndimx, float ystep, float ymin, float ymax){
    //printf("%f, %f\\n",array_x[0],)
    if(x <= array_x[0])return ymin + 0.5*ystep;
    else if(x>array_x[ndimx-1]){printf("%f, %f\\n",x,array_x[ndimx-1]);return ymax+0.5*ystep;}
    else{
            int i;
            for(i = 0; i < ndimx ; i++){
                if(x > array_x[i]);
                else break;
            }
            float y1 = (i) * ystep + ymin + 0.5 * ystep;
            //if(y1>1.004)printf("%d, %f, %f, %f\\n", i, array_x[i-1], array_x[i], y1);
            return y1;
        }
}

__device__ float legendreSeries(float xx, float *par ){
    //ATTENTION:normX is energymax, please remember to change it once the input fitting energy range is different//
    float normX = 150.;
    float fluc = 1. * ( par[0] + par[1] * xx / normX + par[2] * (0.5*(3*xx*xx/normX/normX - 1.)) );
    return fluc;
}


__device__ float legendreSeries_er(float xx, float *par ){
    float normX = par[0];
    float fluc = 1. * ( par[1] + par[2] * xx / normX + + par[3] * (0.5*(3*xx*xx/normX/normX - 1.)));
    return fluc;
}

__device__ float get_tritium_energy_weight(float energy){

    float flat = 30.;//temp results from previous er fitting test, probably need to be modified for later 
    float T = energy; 
    float Q = 18.5906, pi = 3.1415926; 
    if (T > Q||T < 0)return flat; 
    float m = 0.511e3; 
    float P = sqrt(2 * m * T);  
    float E = T + m; 
    float eta = 2.*1./137*E/P; 
    float F = 2*pi*eta/(1-exp(-2*pi*eta)); 
    return float(F * P * E * (Q - T) * (Q - T) * 0.00001 + flat);
}

__device__ float get_dd_energy_weight(float energy){

    float e[] = {100,
                -0.81, 0.81, 2.43, 4.05, 5.67, 7.29, 8.91, 10.53, 12.15, 13.77, 15.39, 17.01, 18.63, 20.25, 21.87, 23.49, 25.11, 26.73, 28.35, 29.97, 31.59, 33.21, 34.83, 36.45, 38.07, 39.69, 41.31, 42.93, 44.55, 46.17, 47.79, 49.41, 51.03, 52.65, 54.27, 55.89, 57.51, 59.13, 60.75, 62.37, 63.99, 65.61, 67.23, 68.85, 70.47, 72.09, 73.71, 75.33, 76.95, 78.57, 80.19, 81.81, 83.43, 85.05, 86.67, 88.29, 89.91, 91.53, 93.15, 94.77, 96.39, 98.01, 99.63, 101.25, 102.87, 104.49, 106.11, 107.73, 109.35, 110.97, 112.59, 114.21, 115.83, 117.45, 119.07, 120.69, 122.31, 123.93, 125.55, 127.17, 128.79, 130.41, 132.03, 133.65, 135.27, 136.89, 138.51, 140.13, 141.75, 143.37, 144.99, 146.61, 148.23, 149.85, 151.47, 153.09, 154.71, 156.33, 157.95, 159.57};
    float weight[] = {100,
                    0, 33349, 92689, 54807, 35523, 25985, 21164, 17741, 15022, 13549, 12105, 11212, 10430, 9919, 9098, 8727, 8004, 7328, 7090, 6800, 6877, 7043, 7191, 7161, 7539, 7674, 7820, 8482, 9017, 9776, 10215, 11533, 12833, 13894, 15288, 16794, 18554, 21013, 23636, 27579, 31118, 29511, 14913, 5499, 4176, 3497, 2941, 2311, 1858, 1637, 1282, 1098, 942, 818, 672, 582, 490, 477, 377, 356, 308, 263, 254, 232, 227, 220, 197, 163, 173, 174, 165, 166, 144, 142, 113, 121, 126, 106, 86, 87, 59, 46, 46, 31, 26, 18, 13, 14, 15, 11, 5, 6, 2, 0, 0, 0, 0, 0, 0, 0};

    return interpolate1d(energy, e, weight);
}

__device__ float get_z_weight(double z){
    float z_lower = -8116.2;
    float z_real = z_lower + z;
    float z_value[] = {100,
                    -8126.3, -8113.7, -8101.1, -8088.5, -8075.9, -8063.3, -8050.7, -8038.1, -8025.5, -8012.9, -8000.3, -7987.7, -7975.1, -7962.5, -7949.9, -7937.3, -7924.7, -7912.1, -7899.5, -7886.9, -7874.3, -7861.7, -7849.1, -7836.5, -7823.9, -7811.3, -7798.7, -7786.1, -7773.5, -7760.9, -7748.3, -7735.7, -7723.1, -7710.5, -7697.9, -7685.3, -7672.7, -7660.1, -7647.5, -7634.9, -7622.3, -7609.7, -7597.1, -7584.5, -7571.9, -7559.3, -7546.7, -7534.1, -7521.5, -7508.9, -7496.3, -7483.7, -7471.1, -7458.5, -7445.9, -7433.3, -7420.7, -7408.1, -7395.5, -7382.9, -7370.3, -7357.7, -7345.1, -7332.5, -7319.9, -7307.3, -7294.7, -7282.1, -7269.5, -7256.9, -7244.3, -7231.7, -7219.1, -7206.5, -7193.9, -7181.3, -7168.7, -7156.1, -7143.5, -7130.9, -7118.3, -7105.7, -7093.1, -7080.5, -7067.9, -7055.3, -7042.7, -7030.1, -7017.5, -7004.9, -6992.3, -6979.7, -6967.1, -6954.5, -6941.9, -6929.3, -6916.7, -6904.1, -6891.5, -6878.9};
    float weight[] = {100,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 8349, 8498, 8387, 8328, 8372, 8327, 8468, 8471, 8594, 8862, 8899, 9219, 9434, 9690, 10028, 10376, 10707, 11339, 11933, 12462, 13489, 14648, 18686, 29819, 35890, 38522, 38785, 38850, 37897, 35834, 27616, 15113, 13109, 12179, 11047, 10260, 9544, 9042, 8400, 7923, 7361, 6903, 6637, 6171, 5853, 5551, 5319, 4939, 4797, 4419, 4132, 4071, 3880, 3808, 3522, 3372, 3240, 3079, 2879, 2821, 2664, 2593, 2612, 2490, 2456, 2302, 2379, 2183, 2158, 2146, 2056, 2056, 1919, 1977, 1793, 1844, 1818, 1766, 1776, 1803, 1775, 1887, 1918, 2021, 233, 0, 0, 0, 0, 0, 0};
    return interpolate1d(z_real, z_value, weight)/758775.;
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
        float L = (Nq / energy) * Wq_eV * 1e-3;
        pars[0] = Wq_eV;
        pars[1] = L;
        pars[2] = NexONi;
        pars[3] = rmean + legendreSeries(energy, &free_pars[4]);
        pars[4] = omega*free_pars[3];
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
      //      float beta decay has density higher than 3g/cc;
        float Qy = QyLvlmedE +
                  (QyLvllowE - QyLvlmedE) /
                      powf(1. + 1.304 * powf(energy, 2.1393), 0.35535) +
                  QyLvlhighE / (1. + DokeBirks * powf(energy, LET_power));
        if (Qy > QyLvllowE && energy > 1. && E_drift > 1e4) Qy = QyLvllowE;
        float Ly = Nq / energy - Qy;
        if(Ly < 0.0)Ly = 0.0;
        if(Qy < 0.0)Qy = 0.0;
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
        //er best-fit pars//
        float er_pol_pars[] = {70., 0.28899712, -0.28359538, 0.42841342};
        pars[0] = Wq_eV;
        pars[1] = 1.;
        pars[2] = NexONi;
        pars[3] = rmean + legendreSeries_er(energy, &er_pol_pars[0]);
        pars[4] = omega;
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
        {11,
        1789.6, 1409.8, 1409.8, 1342.5, 925.15, 1066.1, 901.68, 874.54, 874.39, 851.19, 808.32};
     float duration[] = 
        {11,
        10508.8,  5367. , 47238. ,  2637. , 45605.9,  9036. , 24968. ,45767. , 43316. , 47676. , 27285.};
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

__device__ float get_bls(float * bls_array, float x, curandState_t *rand_state){
    
    if(x<2.)return 1.;
    float xstep = 1.;
    float xmin = 0.;
    float xmax = 200.;
    int ndimx = (int)((xmax-xmin)/xstep);
    float ystep = 0.008;
    float ymin = 0.4;
    float ymax = 1.2;
    int ndimy = (int)((ymax-ymin)/ystep);

    //from s1 value to get hist x bin number//
    int xbin;
    if(x > xmax)xbin = ndimx;
    else xbin = (int) ((x - xmin)/xstep) + 1;  
    
    //get index beginner in bls_array//
    int index_begin = (xbin - 1)* 100; 
    if(index_begin>=ndimx*ndimy)index_begin = ndimx*ndimy-ndimy;

    //get random and find its corresponding y value//
    float dice = curand_uniform(rand_state);
    float y = interpolate1d_bin(dice, &bls_array[index_begin], ndimy, ystep, ymin, ymax);
   // if(y>1.004)printf("%f, %f, %f\\n",dice, x, y);
    return y;
}

__device__ float get_chargeloss(float * chargeloss_array, float x, curandState_t *rand_state){
    if(x<30.)return 1.;
    float xstep = 10.;
    float xmin = 0.;
    float xmax = 1000;
    int ndimx = (int)((xmax-xmin)/xstep);
    float ystep = 0.01;
    float ymin = 0.1;
    float ymax = 1.1;
    int ndimy = (int)((ymax-ymin)/ystep);

    //from s2 value to get hist x bin number//
    int xbin;
    if(x > xmax)xbin = ndimx;
    else xbin = (int) ((x - xmin)/xstep) + 1;
    
    //get index beginner in chargeloss_array//
    int index_begin = (xbin - 1)* 100;
    if(index_begin>=ndimx*ndimy)index_begin = ndimx*ndimy-ndimy;

    //get random and find its corresponding y value//
    float dice =  curand_uniform(rand_state);
    float y = interpolate1d_bin(dice, &chargeloss_array[index_begin], ndimy, ystep, ymin, ymax);
    return y;
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
    float mu = -27.77622479;
    float sigma = 15.83855102;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__device__ float get_cs2b_qualitycut_eff(float x){
    float mu = -366.18230685;
    float sigma = 288.40437576;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__device__ float get_cs1_qualitycut_eff_er(float x){
    float mu = -1.25143721;
    float sigma = 4.22550541;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__device__ float get_cs2b_qualitycut_eff_er(float x){
    float mu = 162.46632354;
    float sigma = 93.36755591;
    return 1 / (expf(-(x-mu)/sigma) + 1);
}

__global__ void signal_simulation(
        int *seed,
        float *input,
        float *nuisance_par,
        float *bls,
        float *chargeloss,
        double *output)
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
    bool simuTypeNR = true;
    int typeFlag = 0;

    //get energy randomly
    float lower = 0.1;
    float upper = 150.;
    
    float energy = curand_uniform(&s)*(upper-lower)+lower;
    float weight = get_dd_energy_weight(energy);
    if(weight<=0.)weight = 0.;
    //get detector parameters

    float g1 = nuisance_par[0]; //pe per photon
    float g2 = nuisance_par[1];//pe per electron(now bottom only)
    float sPEres = 0.3; //pmt single phe resolution (gaussian assumed)
    float P_dphe = 0.22; //probab that 1 phd makes 2nd phe 
    float seg = nuisance_par[2]; //single electron gain, num of pe/electron(now bottom only)
    
    //ATTERNTION: This two pars are totally in hit/electron!!!//
    float SEG = seg/(1.+ P_dphe);
    float G1 = g1/(1. + P_dphe);

    float ExtraEff = g2/seg ; //electron extraction eff
    float deltaG = 0.27 * SEG; //SEG resolution
    float eLife_us = get_elife(&s,typeFlag); //the drift electron in microsecond
    
    //float hit_thr = 1.;
    float s1_thr = 2.;
    float s2b_thr_beforecorrection = 80*0.27;
    float s2b_thr_recluster = 30.;
    float s1_max = 120;
    //float s2_max = 20000;
    float E_drift = 115.6; //drift electric field in V/cm
    float driftvelocity = 1.2; //in mm/microsecond
    float dt_min = 20;//in ms
    float dt_max = 770;//in ms
    float zup = 1200. - dt_min * driftvelocity;
    float zdown = 1200. - dt_max * driftvelocity;
    float density = 2.8611; //density of liquid xenon
    float TopDrift = 1200.; // mm 
    
    float pars[5] = {0.};//={W_eV,Nex/Ni, rmean, rdelta}
    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0], &nuisance_par[0]);
    float w_eV = pars[0];
    float L = pars[1];
    float NexONi = pars[2];
    float rmean = pars[3];
    float deltaR = pars[4];
    if(rmean<0.)return;
    else if(rmean>1.)return;

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
    float r = gpu_truncated_gaussian(&s, rmean, deltaR, 0., 1.);
    if(r<0.)return;
    else if(r>1.)return;

    // 6) get photon and electron numbers
    int Ne = gpu_binomial(&s, Ni, 1 - r);
    int Nph = Ni + Nex - Ne;

    // 7) get drift time 
    float truthz = zdown + curand_uniform(&s) * (zup - zdown);
    weight *= get_z_weight(truthz);
    float dt = (TopDrift - truthz) / driftvelocity;

    // 8) get g1 hit(phd) number
    float g1_true = get_g1_true(dt,G1);
    int nHitsS1 = gpu_binomial(&s, Nph, g1_true);
    
    // 9) get s1 in phe #, consider float phe
    int NpheS1 = nHitsS1 + gpu_binomial(&s, nHitsS1, P_dphe);
    
    // 10) s1 pulse area, with pmt resolution
    float pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    //if(pulseAreaS1 <= 0.)return;
    if(pulseAreaS1 <= 0.)pulseAreaS1 = 0.;

    // 11) biased s1 pulse area (same in s2 bias)
    float biasS1 = get_bls(&bls[0], pulseAreaS1, &s);
    float pulseAreaBiasS1 = pulseAreaS1 * biasS1;
    //if(pulseAreaBiasS1 <= 0.)return;
    if(pulseAreaBiasS1 <= 0.)pulseAreaBiasS1 = 0.;

    // 12) corrected s1 pulse area
    float InversedS1Correction = get_g1_inverse_correction_factor(dt);
    float pulseAreaS1Cor = pulseAreaBiasS1 * InversedS1Correction;
    //if(pulseAreaS1Cor <= 0.)pulseAreaS1Cor = 0.;
    //if(pulseAreaS1Cor <= s1_thr)return;
    if(pulseAreaS1Cor <= s1_thr||pulseAreaS1Cor > s1_max)return;

    // 13) do electron drifting and extraction
    int Nee = gpu_binomial(&s, Ne, expf(-dt / eLife_us) * ExtraEff);

    // 14) get s2 hit number 
    int nHitsS2b = (int)(curand_normal(&s)*sqrtf((float)Nee)*deltaG + SEG*(float)Nee);
    if(nHitsS2b <= 0.)return;
    //if(nHitsS2b <= 0.)nHitsS2b = 0.;

    // 15) get s2 phe
    int NpheS2b = nHitsS2b + gpu_binomial(&s, nHitsS2b, P_dphe);
    
    // 16) s2 pulse area, with pmt resolution
    float pulseAreaS2b = curand_normal(&s)*sqrtf(NpheS2b)*sPEres + NpheS2b; 
    //if(pulseAreaS2b <= 0.)pulseAreaS2b = 0.;
    if(pulseAreaS2b < s2b_thr_recluster)return;

    // 17) biased s2 pulse area 
    float biasS2 = get_chargeloss(&chargeloss[0], pulseAreaS2b, &s);
    float pulseAreaBiasS2b = pulseAreaS2b * biasS2;
    //float pulseAreaBiasS2b = pulseAreaS2b * 1.;
    if(pulseAreaBiasS2b <= s2b_thr_beforecorrection)return;
    //if(pulseAreaBiasS2 <=0.)pulseAreaBiasS2 = 0.;

    // 18）corrected s2 pulse area
    float InversedS2Correction = get_g2_inverse_correction_factor(dt, eLife_us);
    float pulseAreaS2Corb = pulseAreaBiasS2b * InversedS2Correction;    
    //if(pulseAreaS2Corb <= s2_thr)return;
    //if(pulseAreaS2Corb < s2b_thr_beforecorrection)return;

    weight *= get_s1_tagging_eff(simuTypeNR, pulseAreaS1Cor);
    weight *= get_cs1_qualitycut_eff(pulseAreaS1Cor); 
    weight *= get_cs2b_qualitycut_eff(pulseAreaS2Corb);
    
    //printf("%f ,%f, %f, %f\\n",eLife_us, pulseAreaS1Cor, pulseAreaS2, weight); 
    int xBinning = (int)*(input+1);
    float xMin = (float)*(input+2);
    float xMax = (float)*(input+3);
    float xStep = (xMax - xMin)/(float)xBinning;
    int yBinning = (int)*(input+4);
    float yMin = (float)*(input+5);
    float yMax = (float)*(input+6);
    float yStep = (yMax - yMin)/(float)yBinning;

    atomicAdd(&output[0],(double)1.);
    //get values, overflows and underflows in output[1]-output[9]
    float xvalue = (float)pulseAreaS1Cor;
    float yvalue = (float)log10f(pulseAreaS2Corb);
    if(xvalue<xMin && yvalue>=yMax)atomicAdd(&output[2],(double)1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue>=yMax)atomicAdd(&output[3], (double)1.);
    else if(xvalue>=xMax && yvalue>=yMax)atomicAdd(&output[4], (double)1.);
    else if(xvalue<xMin && yvalue>yMin && yvalue<yMax)atomicAdd(&output[5], (double)1.);
    else if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
    {
        int xbin = (int) ((xvalue - xMin)/xStep) + 1; 
        int ybin = (int) ((yvalue - yMin)/yStep) + 1;
        atomicAdd(&output[6], (double)1.);
        atomicAdd(&output[1],(double)weight);
        int index = 10+(ybin-1)*xBinning+xbin;
        int errindex = 10+xBinning*yBinning+((ybin-1)*xBinning+xbin);
        atomicAdd(&output[index], (double)weight);
        atomicAdd(&output[errindex],(double)(weight*weight));
    }
    else if(xvalue>=xMax && yvalue>yMin && yvalue<yMax)atomicAdd(&output[7], (double)1.);
    else if(xvalue<xMin && yvalue<yMin)atomicAdd(&output[8], (double)1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue<yMin)atomicAdd(&output[9], (double)1.);
    else if(xvalue>=xMax && yvalue<yMin)atomicAdd(&output[10], (double)1.);

}

__global__ void signal_simulation_er(
        int *seed,
        float *input,
        float *nuisance_par,
        float *bls,
        float *chargeloss,
        double *output)
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
    float weight = get_tritium_energy_weight(energy);
    if(weight<=0.)weight = 0.;
    weight *= nuisance_par[7];

    //get detector parameters

    float g1 = nuisance_par[0]; //pe per photon
    float g2 = nuisance_par[1];//pe per electron(now bottom only)
    float sPEres = 0.3; //pmt single phe resolution (gaussian assumed)
    float P_dphe = 0.22; //probab that 1 phd makes 2nd phe 
    float seg = nuisance_par[2]; //single electron gain, num of pe/electron(now bottom only)
    
    //ATTERNTION: This two pars are totally in hit/electron!!!//
    float SEG = seg/(1.+ P_dphe);
    float G1 = g1/(1. + P_dphe);

    float ExtraEff = g2/seg ; //electron extraction eff
    float deltaG = 0.27 * SEG; //SEG resolution
    float eLife_us = get_elife(&s,typeFlag); //the drift electron in microsecond
    
    //float hit_thr = 1.;
    float s1_thr = 2.;
    float s2b_thr_beforecorrection = 80*0.27;
    float s2b_thr_recluster = 30.;
    //float s2b_thr = 80*0.25; //s2 threshold in phe
    float s1_max = 120;
    //float s2_max = 20000;
    float E_drift = 115.6; //drift electric field in V/cm
    float driftvelocity = 1.2; //in mm/microsecond
    float dt_min = 20;//in ms
    float dt_max = 770;//in ms
    float zup = 1200. - dt_min * driftvelocity;
    float zdown = 1200. - dt_max * driftvelocity;
    float density = 2.8611; //density of liquid xenon
    float TopDrift = 1200.; // mm 
    
    float pars[5] = {0.};//={W_eV,Nex/Ni, rmean, rdelta}
    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0], &nuisance_par[0]);
    float w_eV = pars[0];
    float L = pars[1];
    float NexONi = pars[2];
    float rmean = pars[3];
    float deltaR = pars[4];
    if(rmean<0.)return;
    else if(rmean>1.)return;

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
    float r = gpu_truncated_gaussian(&s, rmean, deltaR, 0., 1.);
    if(r<0.)return;
    else if(r>1.)return;

    // 6) get photon and electron numbers
    int Ne = gpu_binomial(&s, Ni, 1 - r);
    int Nph = Ni + Nex - Ne;

    // 7) get drift time 
    float truthz = zdown + curand_uniform(&s) * (zup - zdown);
    float dt = (TopDrift - truthz) / driftvelocity;

    // 8) get g1 hit(phd) number
    float g1_true = get_g1_true(dt,G1);
    int nHitsS1 = gpu_binomial(&s, Nph, g1_true);
    
    // 9) get s1 in phe #, consider float phe
    int NpheS1 = nHitsS1 + gpu_binomial(&s, nHitsS1, P_dphe);
    
    // 10) s1 pulse area, with pmt resolution
    float pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    //if(pulseAreaS1 <= 0.)return;
    if(pulseAreaS1 <= 0.)pulseAreaS1 = 0.;

    // 11) biased s1 pulse area (same in s2 bias)
    float biasS1 = get_bls(&bls[0], pulseAreaS1, &s);
    float pulseAreaBiasS1 = pulseAreaS1 * biasS1;
    //if(pulseAreaBiasS1 <= 0.)return;
    if(pulseAreaBiasS1 <= 0.)pulseAreaBiasS1 = 0.;

    // 12) corrected s1 pulse area
    float InversedS1Correction = get_g1_inverse_correction_factor(dt);
    float pulseAreaS1Cor = pulseAreaBiasS1 * InversedS1Correction;
    //if(pulseAreaS1Cor <= 0.)pulseAreaS1Cor = 0.;
    //if(pulseAreaS1Cor <= s1_thr)return;
    if(pulseAreaS1Cor <= s1_thr||pulseAreaS1Cor > s1_max)return;

    // 13) do electron drifting and extraction
    int Nee = gpu_binomial(&s, Ne, expf(-dt / eLife_us) * ExtraEff);

    // 14) get s2 hit number 
    int nHitsS2b = (int)(curand_normal(&s)*sqrtf((float)Nee)*deltaG + SEG*(float)Nee);
    if(nHitsS2b <= 0.)return;
    //if(nHitsS2b <= 0.)nHitsS2b = 0.;

   // 15) get s2 phe
    int NpheS2b = nHitsS2b + gpu_binomial(&s, nHitsS2b, P_dphe);
    
    // 16) s2 pulse area, with pmt resolution
    float pulseAreaS2b = curand_normal(&s)*sqrtf(NpheS2b)*sPEres + NpheS2b; 
    //if(pulseAreaS2b <= 0.)pulseAreaS2b = 0.;
    if(pulseAreaS2b < s2b_thr_recluster)return;

    // 17) biased s2 pulse area 
    float biasS2 = get_chargeloss(&chargeloss[0], pulseAreaS2b, &s);
    float pulseAreaBiasS2b = pulseAreaS2b * biasS2;
    //float pulseAreaBiasS2b = pulseAreaS2b * 1.;
    if(pulseAreaBiasS2b <= s2b_thr_beforecorrection)return;
    //if(pulseAreaBiasS2 <=0.)pulseAreaBiasS2 = 0.;

    // 18）corrected s2 pulse area
    float InversedS2Correction = get_g2_inverse_correction_factor(dt, eLife_us);
    float pulseAreaS2Corb = pulseAreaBiasS2b * InversedS2Correction;    
    //if(pulseAreaS2Corb <= s2_thr)return;
    //if(pulseAreaS2Corb < s2b_thr_beforecorrection)return;
    
    weight *= get_s1_tagging_eff(true, pulseAreaS1Cor);
    weight *= get_cs1_qualitycut_eff(pulseAreaS1Cor); 
    weight *= get_cs2b_qualitycut_eff(pulseAreaS2Corb);
    
    //printf("%f ,%f, %f, %f\\n",eLife_us, pulseAreaS1Cor, pulseAreaS2, weight); 
    int xBinning = (int)*(input+1);
    float xMin = (float)*(input+2);
    float xMax = (float)*(input+3);
    float xStep = (xMax - xMin)/(float)xBinning;
    int yBinning = (int)*(input+4);
    float yMin = (float)*(input+5);
    float yMax = (float)*(input+6);
    float yStep = (yMax - yMin)/(float)yBinning;

    atomicAdd(&output[0],(double)1.);
    //get values, overflows and underflows in output[1]-output[9]
    float xvalue = (float)pulseAreaS1Cor;
    float yvalue = (float)log10f(pulseAreaS2Corb);
    //printf("%d, %f", iteration, xvalue);
    if(xvalue<xMin && yvalue>=yMax)atomicAdd(&output[2],(double)1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue>=yMax)atomicAdd(&output[3], (double)1.);
    else if(xvalue>=xMax && yvalue>=yMax)atomicAdd(&output[4], (double)1.);
    else if(xvalue<xMin && yvalue>yMin && yvalue<yMax)atomicAdd(&output[5], (double)1.);
    else if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
    {
        int xbin = (int) ((xvalue - xMin)/xStep) + 1;
        int ybin = (int) ((yvalue - yMin)/yStep) + 1;
        atomicAdd(&output[6], (double)1.);
        atomicAdd(&output[1],(double)weight);
        int index = 10+(ybin-1)*xBinning+xbin;
        int errindex = 10+xBinning*yBinning+((ybin-1)*xBinning+xbin);
        atomicAdd(&output[index], (double)weight);
        atomicAdd(&output[errindex],(double)(weight*weight));
    }
    else if(xvalue>=xMax && yvalue>yMin && yvalue<yMax)atomicAdd(&output[7], (double)1.);
    else if(xvalue<xMin && yvalue<yMin)atomicAdd(&output[8], (double)1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue<yMin)atomicAdd(&output[9], (double)1.);
    else if(xvalue>=xMax && yvalue<yMin)atomicAdd(&output[10], (double)1.);
}
}
"""
