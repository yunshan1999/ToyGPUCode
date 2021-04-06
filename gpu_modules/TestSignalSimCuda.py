# #############################################################
# Signal simulation details shall be written here!
# this currently is just a test and demonstration!
# #############################################################
# Created @ 2021-03-16
# by Qing Lin
# email: qinglin@ustc.edu.cn
###############################################################

pandax4t_signal_sim = """
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
extern "C" {
__device__ float curand_uniform_1stcall(curandState_t *rand_state){
    float rand = curand_uniform(rand_state);
    return rand;
}

__device__ float curand_uniform_2ndcall(curandState_t *rand_state){
    float rand0 = curand_uniform(rand_state);
    float rand1 = curand_uniform_1stcall(rand_state);
    return rand1;
}

__device__ float curand_uniform_3rdcall(curandState_t *rand_state){
    float rand0 = curand_uniform(rand_state);
    float rand1 = curand_uniform_1stcall(rand_state);
    float rand2 = curand_uniform_2ndcall(rand_state);
    return rand2;
}

__device__ float curand_normal_1stcall(curandState_t *rand_state){
    float rand = curand_normal(rand_state);
    return rand;
}

__device__ float curand_normal_2ndcall(curandState_t *rand_state){
    float rand0 = curand_normal(rand_state);
    float rand1 = curand_normal_1stcall(rand_state);
    return rand1;
}

__device__ float curand_normal_3rdcall(curandState_t *rand_state){
    float rand0 = curand_normal(rand_state);
    float rand1 = curand_normal_1stcall(rand_state);
    float rand2 = curand_normal_2ndcall(rand_state);
    return rand2;
}
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

__device__ float get_er_energy_weight(float energy){

    float Energy[] = {6,
                    7.2067117462943315,
                    33.92045118998408 ,
                    53.04914958955885 ,
                    79.78134412094721 ,
                    98.85467725742637 ,
                    121.72053091596308
                    };
    float Rate[] = {6,
                0.04711112624689215,
                0.050370838075343045,
                0.057582505387584616,
                0.06366109318181677,
                0.06582667776445614,
                0.06582667776445614
                };
    float Weight = interpolate1d(energy, Energy, Rate);
    return Weight;
}

__device__ float get_nr_energy_weight(float energy){

    float Energy[] = {25,
                        0.3643724696356276, 0.9311740890688256,1.4979757085020236,2.145748987854251,
                        2.793522267206477,3.279352226720647, 3.9271255060728745,4.574898785425098,
                        5.222672064777328, 5.789473684210527, 6.356275303643727, 6.842105263157894,
                        7.489878542510123, 8.137651821862349, 8.785425101214575, 12.429149797570851,
                        15.263157894736839,20.202429149797567,26.194331983805668,30.32388663967611, 
                        35.4251012145749,  40.445344129554655,45.87044534412956, 50.89068825910931, 
                        54.93927125506073
                        };
                
    float Rate[] = {25,
                    0.001348559553052562,0.0005837592378488603, 0.00034077484777388155,0.0002380254399901921,
                    0.00017130349897073656,0.00013485595530525607,0.00010616320618413007,0.00008611224963143029,
                    0.00006984830058471098,0.000060147942962818105,0.000050268821214294545,0.00004201231599618962,
                    0.00003727593720314938,0.000030235661912054828,0.000026826957952797274,0.000010938582306368928,
                    0.000005665611008252431,0.0000019306977288832498,7.640413849058564e-7,4.2012315996189623e-7,
                    1.7650346953636906e-7,5.336699231206302e-8,6.014794296281786e-8,3.407748477738809e-8,
                    1.6625672479242957e-8};

    float Weight = interpolate1d(energy, Energy, Rate);
    return Weight;
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

__device__ float get_lindhard_factor(bool simuTypeNR, float energy){
    if(simuTypeNR){
        float epsilon = 11.5 * energy * powf(54, -7./3.);
        float g = 3 * powf(epsilon, 0.15) + 0.7 * powf(epsilon, 0.6) + epsilon;
        float kappa = 0.1394;
        float L = kappa * g / (1 + kappa * g);
        return L;
    }
    else{
        return 1.;
    }
}

__device__ float get_exciton_ratio(bool simuTypeNR, float E_drift, float energy, float density){
    if(simuTypeNR){
        float epsilon = 11.5 * energy * powf(54, -7./3.);
        float alpha = 1.24;
        float zeta = 0.0472;
        float beta = 239.;
        float NexONi = alpha * powf(E_drift,-zeta) * (1 - expf(-beta * epsilon));
        return NexONi;
    }
    else{
        float alpha = 0.067366 + density * 0.039693;
        float NexONi = alpha * erff(0.05 * energy);
        return NexONi;
    }
}

__device__ float get_recomb_frac(bool simuTypeNR, float E_drift, float Ni, float energy){
    if(simuTypeNR){
        float gamma = 0.01385;
        float delta = 0.0620;
        float sigma = gamma * powf(E_drift, -delta);
        float rmean = 1. - logf(1 + Ni * sigma )/(Ni * sigma );
        return rmean;
    }
    else{
        float gamma = 0.124;
        float omega = 31.;
        float delta = 0.24;
        float q0 = 1.13;
        float q1 = 0.47;
        float sigma = gamma * expf(-energy / omega) * powf(E_drift, -delta);
        float rmean = (1. - logf(1 + Ni * sigma/4. )/(Ni * sigma/4.))/(1 + expf(-(energy - q0)/q1));
        return rmean;
    }
}

__device__ float get_recomb_frac_delta(float energy){
    float q2 = 0.04;
    float q3 = 1.7;
    float deltaR = q2 * (1 - expf(-energy/q3));
    return deltaR;
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

    //get energy randomly
    float lower = 0.;
    float upper = 10.;
    
    float energy = curand_uniform(&s)*(upper-lower)+lower;
    float weight = get_er_energy_weight(energy);
    //if(weight<=0.)weight = 0.;
    
    //get detector parameters

    float g1 = nuisance_par[0]; //hit per photon
    float sPEres = nuisance_par[1]; //pmt single phe resolution (gaussian assumed)
    float P_dphe = nuisance_par[2]; //probab that 1 phd makes 2nd phe 
    float SEG = nuisance_par[3]; //single electron gain, num of photon/electron 
                                        //before elife decay&EEE
    float g2 = nuisance_par[4]; //phd/electron
    float ExtraEff = nuisance_par[5]; //electron extraction eff
    float deltaG = nuisance_par[6]; //SEG resolution
    float eLife_us = nuisance_par[7]; //the drift electron in microsecond
    
    //float s2_thr = 80.; //s2 threshold in phe
    float E_drift = 114.2; //drift electric field in V/cm
    float driftvelocity = 1.37824; //in mm/microsecond
    float density = 2.8611; //density of liquid xenon
    float TopDrift = 1200.; // mm 
    float w_eV = 13.7; //average energy of each quanta in eV

    //let the simulation begins

    // 1) get mean quanta number
    int Nq_mean = (int)(energy / w_eV * 1000.);
    
    // 2) get actual quanta number by fluctuating quanta number with fano factor
    float Fano = get_fano_factor(simuTypeNR, density, Nq_mean, E_drift);
    int Nq_actual = (int)(curand_normal(&s)*sqrtf(Fano * Nq_mean) + Nq_mean);
    if(Nq_actual <= 0. )Nq_actual = 0.;
    
    // 3) get quanta number after lindhard factor fluctuation
    float L = get_lindhard_factor(simuTypeNR, energy);
    int Nq = gpu_binomial(&s, Nq_actual, L);
    if(Nq <= 0.)Nq = 0.;
    
    // 4）get exciton ratio and do fluctuation
    float NexONi = get_exciton_ratio(simuTypeNR, E_drift, energy, density);
    if(NexONi < 0.||NexONi > 1.)printf("error in NexONi");
    int Ni = gpu_binomial(&s, Nq, 1/(1 +NexONi));
    int Nex = Nq - Ni;

    // 5) get recomb fraction fluctuation
    float rmean = get_recomb_frac(simuTypeNR, E_drift, Ni, energy);
    float deltaR = get_recomb_frac_delta(energy);
    float r = curand_normal(&s)*deltaR + rmean;
    if(r >= 1. )r = 1.;
    else if(r <= 0.)r = 0.;

    // 6) get photon and electron numbers
    int Ne = gpu_binomial(&s, Ni, 1 - r);
    int Nph = Ni + Nex - Ne;

    // 7) get drift time 
    float truthz = curand_uniform(&s) * TopDrift;
    float dt = (TopDrift - truthz) / driftvelocity;

    // 8) get g1 hit(phd) number
    float g1_true = get_g1_true(dt,g1);
    int nHitsS1 = gpu_binomial(&s, Nph, g1_true);
    //if(nHitsS1 <= 0.)return;
    if(nHitsS1 <= 0.)nHitsS1 = 0.;
    
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
    //if(pulseAreaS1Cor <= 0.)return;
    if(pulseAreaS1Cor <= 0.)pulseAreaS1Cor = 0.;

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
    //if(pulseAreaS2 <= 0.)return;
    if(pulseAreaS2 <= 0.)pulseAreaS2 = 0.;

    // 17) biased s2 pulse area 
    float biasS2, fluctuationS2;
    get_bls(pulseAreaS2, &biasS2, &fluctuationS2);
    float pulseAreaBiasS2 = pulseAreaS2 * (curand_normal(&s)*fluctuationS2 + biasS2);
   // if(pulseAreaBiasS2 <=0.)return;
    if(pulseAreaBiasS2 <=0.)pulseAreaBiasS2 = 0.;

    // 18）corrected s2 pulse area
    float InversedS2Correction = get_g2_inverse_correction_factor(dt, eLife_us);
    float pulseAreaS2Cor = pulseAreaBiasS2 * InversedS2Correction;    
    //if(pulseAreaS2Cor <= 0.)return;
    if(pulseAreaS2Cor <= 0.)pulseAreaS2Cor = 0.;
  
    if(pulseAreaS1Cor <= 0.||pulseAreaS2Cor <= 0.)return;
    
    int xBinning = (int)*(input+1);
    float xMin = (float)*(input+2);
    float xMax = (float)*(input+3);
    float xStep = (xMax - xMin)/(float)xBinning;
    int yBinning = (int)*(input+4);
    float yMin = (float)*(input+5);
    float yMax = (float)*(input+6);
    float yStep = (yMax - yMin)/(float)yBinning;

    atomicAdd(&output[0], 1.);
    //get values, overflows and underflows in output[1]-output[9]
    float xvalue = (float)pulseAreaS1Cor;
    float yvalue = (float)log10f(pulseAreaS2Cor/pulseAreaS1Cor);
    
    if(xvalue<xMin && yvalue>=yMax)atomicAdd(&output[1], 1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue>=yMax)atomicAdd(&output[2], 1.);
    else if(xvalue>=xMax && yvalue>=yMax)atomicAdd(&output[3], 1.);
    else if(xvalue<xMin && yvalue>yMin && yvalue<yMax)atomicAdd(&output[4], 1.);
    else if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
    {
        int xbin = (int) ((xvalue - xMin)/xStep) + 1; 
        int ybin = (int) ((yvalue - yMin)/yStep) + 1;
        weight *= get_s1_efficiency(nHitsS1);
        if(weight<0.){weight = 0.;}
        atomicAdd(&output[5], 1.);
        int index = 9+(ybin-1)*xBinning+xbin;
        int weightindex = 9+xBinning*yBinning+((ybin-1)*xBinning+xbin);
        atomicAdd(&output[index], weight);
        atomicAdd(&output[weightindex],weight*weight);
    }
    else if(xvalue>=xMax && yvalue>yMin && yvalue<yMax)atomicAdd(&output[6], 1.);
    else if(xvalue<xMin && yvalue<yMin)atomicAdd(&output[7], 1.);
    else if(xvalue>xMin && xvalue<xMax && yvalue<yMin)atomicAdd(&output[8], 1.);
    else if(xvalue>=xMax && yvalue<yMin)atomicAdd(&output[9], 1.);
}
}

"""
