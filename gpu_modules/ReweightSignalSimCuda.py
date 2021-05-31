
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

  __device__ float legendreSeries(float xx, float *par )
  {
    float normX = par[0];
    float fluc = 1. * ( par[1] + par[2] * xx / normX + par[3] * (0.5*(3*xx*xx/normX/normX - 1.)) );
    return fluc;
  }
  
  __device__ float get_efficiency_FermiDirac(float xx,float *par){
    float eff = 1./(exp(-1.*(xx-par[0])/par[1]) + 1.);
    return eff;
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
    if(prob_success < 0 || prob_success > 1) return -1.;
    if(prob_success == 0) return 0;
    if(prob_success == 1) return num_trials;
    // do a gaussian approximation if:
    // np(1-p)>10
    if ( ((float)num_trials)*prob_success*(1.-prob_success)>50 )
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


  __device__ int gpu_binomial_truncate(float z, curandState_t *rand_state, int num_trials, float prob_success)
  {
    int obs = gpu_binomial(rand_state,num_trials,prob_success);
    float mean = float(num_trials) * prob_success;
    float sigma = sqrtf(float(num_trials) * prob_success*(1-prob_success));
    
    if(mean < 10.)
    {
      if(sigma > 3.) z *=1.5;
      //else if(sigma > 1.) z *=1.5;
      else if (sigma >0.25) z *= 2.5;
      else if (sigma >0.1) z *= 5.5;
      else  return int(num_trials*prob_success);

    }
    while (float(obs) > mean + z * sigma || float(obs) < mean - z * sigma)
      obs = gpu_binomial(rand_state,num_trials,prob_success);
    return obs;
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


  __device__ float get_tritium_energy_weight(float energy,float *par)
  {
    float flat=par[0];
    float T = energy;
    float Q(18.5906),pi(3.1415926);
    if (T>Q||T<0)  return flat;
    float m = 0.511e3;
    float P = sqrt(2*m*T);
    float E = T+m;
    float eta = 2.*1./137*E/P;
    float F = 2*pi*eta/(1-exp(-2*pi*eta));
    return float(F*P*E*(Q-T)*(Q-T)*0.00001+flat);

  }

  __device__ float get_energy_weight(float energy,float *par,int type)
  {
    if(type == 3 ) return get_tritium_energy_weight(energy,par);
    else if(type == 220 ) return 1.;
  
  }

  __device__ float get_elife(curandState_t *rand_state, float *elife, float *duration){
    float temp = 0.;
    if(int(elife[0]) != int(duration[0]) )return temp;
    float duration_tot = 0.;
    int n = int(elife[0]);
    for(int i = 1; i <= n ; i++){
      duration_tot += (float)duration[i];
    }
    float dice = curand_uniform(rand_state) * duration_tot;
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

  __device__ bool get_yield_pars(bool simuTypeNR, float E_drift, float energy, float density, float * pars){
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
      pars[3] = rmean;
      pars[4] = omega;
      if(pars[3]<=0.) return false;
      if(pars[3]>=1.) return false;
      if(pars[4]<=0.) return false;
      return true;
    }
    else{
      // float Wq_eV = 1.9896 + (20.8 - 1.9896) / (1. + powf(density / 4.0434, 1.4407));

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
      pars[3] = rmean ;
      pars[4] = omega;
      if(pars[3]<=0.) return false;
      if(pars[3]>=1.) return false;
      if(pars[4]<=0.) return false;
      return true;
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
      float *const_pars,
      float *v_variables,
      float *elife, float *elife_duration,
      float *output_data, int stride_output_data, int N_output_data,int mode)
  {
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    curandState s;
    curand_init((*seed)*iteration,2, 0, &s);
    
    const int num_trials = (int)const_pars[0];
    // get the number of trials


    // keep only the GPU nodes that satisfy the number of trial
    if (iteration>num_trials) return;


    //to determine whether it is ER or NR
    bool simuTypeNR = bool(const_pars[1]);

    
    //input detector parameters
    int energyType = int(const_pars[10]);

    float sPEres = const_pars[11] ; //pmt single phe resolution (gaussian assumed)
    float P_dphe = const_pars[12]; //probab that 1 phd makes 2nd phe 
    float E_drift = const_pars[13]; //drift electric field in V/cm
    float driftvelocity = const_pars[14]; //in mm/microsecond
    float dt_min = const_pars[15];//in us
    float dt_max = const_pars[16];//in us
    float TopDrift = const_pars[17]; // drifting z distancemm
    float zup = 1200. - dt_min * driftvelocity;
    float zdown = 1200. - dt_max * driftvelocity;
    float density = 2.8611; //density of liquid xenon

    float eLife_us = get_elife(&s,elife,elife_duration); //the drift electron in microsecond




    //get energy randomly
    float e_lower = (float)const_pars[2];
    float e_upper = (float)const_pars[3];
    float powerE = 0.95;
    float e_lower_log = powf(powerE,e_lower);
    float e_upper_log = powf(powerE,e_upper);

    //mode 0: pool event generation before reweighting
    //mode 1: traditional nest simulation
    float energy;
    if (mode == 0 && powerE != 1. ) energy = logf(curand_uniform(&s)*(e_upper_log-e_lower_log)+e_lower_log)/logf(powerE); // balance out phase space dependence on energy in the pool
    else energy = curand_uniform(&s)*(e_upper - e_lower) + e_lower; 

    while (energy < 0.) energy = curand_uniform(&s)*(e_upper - e_lower) + e_lower;

    //get all fitting parameters including nuissance
    const int stride_v_variables = 8;//4bytes for one float 
    float g1_l = *( (float*) ((char*)v_variables+ 0 * stride_v_variables)); //hit per photon
    float g1_u =*( (float*) ((char*)v_variables+ 0 * stride_v_variables ) + 1); //hit per photon
    
    
    float g2b_l = *( (float*) ((char*)v_variables+ 1 * stride_v_variables)); 
    float g2b_u = *( (float*) ((char*)v_variables+ 1 * stride_v_variables) + 1); 


    float SEGb_l = *( (float*) ((char*)v_variables+ 2 * stride_v_variables)); 
    float SEGb_u = *( (float*) ((char*)v_variables+ 2 * stride_v_variables) + 1); 

    if (energy < 4.) 
    {
      g1_u = 0.5;//a larger fluctuation at low energy
      g1_l = 0.;
      g2b_u *=1.;
      SEGb_u *= 1.;
    }
    bool gaussianNuissance = true;

    float G1;
    float g2b;
    float SEGb;
    float eee;
    float sigma_nui_ratio = 0.33;
    if(mode == 0)
    {
      if(gaussianNuissance)
      {

        G1 = gpu_truncated_gaussian(&s,(g1_u+g1_l)*0.5,(g1_u-g1_l)*sigma_nui_ratio,g1_l,g1_u);
        g2b = gpu_truncated_gaussian(&s,(g2b_u+g2b_l)*0.5,(g2b_u-g2b_l)*sigma_nui_ratio,g2b_l,g2b_u);
        SEGb = gpu_truncated_gaussian(&s,(SEGb_u+SEGb_l)*0.5,(SEGb_u-SEGb_l)*sigma_nui_ratio,SEGb_l,SEGb_u);
        eee = g2b / SEGb;
        while(eee > 1. )
        {
          g2b = gpu_truncated_gaussian(&s,(g2b_u+g2b_l)*0.5,(g2b_u-g2b_l)*0.25,g2b_l,g2b_u);
          SEGb = gpu_truncated_gaussian(&s,(SEGb_u+SEGb_l)*0.5,(SEGb_u-SEGb_l)*0.25,SEGb_l,SEGb_u);
          eee = g2b / SEGb;
        }
      }
      else{
        G1 = curand_uniform(&s)*(g1_u-g1_l)+g1_l;
        g2b = curand_uniform(&s)*(g2b_u-g2b_l)+g2b_l;
        SEGb = curand_uniform(&s)*(SEGb_u-SEGb_l)+SEGb_l;
        eee = g2b / SEGb;//equivalent to only detect S2 bottom
        while(eee > 1. )
        {
          g2b = curand_uniform(&s)*(g2b_u-g2b_l)+g2b_l;
          SEGb = curand_uniform(&s)*(SEGb_u-SEGb_l)+SEGb_l;
          eee = g2b / SEGb;
        }

      }
    }




    float p2Recomb_l = *( (float*) ((char*)v_variables+ 3 * stride_v_variables)); 
    float p2Recomb_u = *( (float*) ((char*)v_variables+ 3 * stride_v_variables) + 1); 
    float p2Recomb = curand_uniform(&s)*(p2Recomb_u-p2Recomb_l)+p2Recomb_l;

    float p0Recomb_l = *( (float*) ((char*)v_variables+ 4 * stride_v_variables)); 
    float p0Recomb_u = *( (float*) ((char*)v_variables+ 4 * stride_v_variables) + 1); 
    float p0Recomb = curand_uniform(&s)*(p0Recomb_u-p0Recomb_l)+p0Recomb_l;
    
    float p1Recomb_l = *( (float*) ((char*)v_variables+ 5 * stride_v_variables)); 
    float p1Recomb_u = *( (float*) ((char*)v_variables+ 5 * stride_v_variables) + 1); 
    float p1Recomb = curand_uniform(&s)*(p1Recomb_u-p1Recomb_l)+p1Recomb_l;


    float p0FlucRecomb_l = *( (float*) ((char*)v_variables+ 6 * stride_v_variables)); 
    float p0FlucRecomb_u = *( (float*) ((char*)v_variables+ 6 * stride_v_variables) + 1); 
    float p0FlucRecomb = curand_uniform(&s)*(p0FlucRecomb_u-p0FlucRecomb_l)+p0FlucRecomb_l;

    float flatE_l = *( (float*) ((char*)v_variables+ 7 * stride_v_variables)); 
    float flatE_u = *( (float*) ((char*)v_variables+ 7 * stride_v_variables) + 1); 
    float flatE = curand_uniform(&s)*(flatE_u-flatE_l)+flatE_l;
    
    bool debug = false;
    
    if(mode == 1)
    {
      G1 = (float) *(v_variables);
      g2b = (float) *(v_variables + 1);
      SEGb = (float) *(v_variables + 2);
      eee = g2b / SEGb;
      p2Recomb = (float) *(v_variables + 3);
      p0Recomb = (float) *(v_variables + 4);
      p1Recomb = (float) *(v_variables + 5);
      p0FlucRecomb = (float) *(v_variables + 6);
      flatE = (float) *(v_variables + 7);

    }


    int max_step = 100;//maximum trials to get rid of unphysical events in a while loop
    
    //float pars[5]={w_eV,lindhard_stretch,NexOverNi,rmean,deltaR};
    float pars[5]={0};
    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0]);
    float w_eV = pars[0];
    float L = pars[1];//lindhard factor
    float NexONi = pars[2];
    
    //random deltaR
    float deltaR = pars[4] * p0FlucRecomb;
    if(debug)
    {
      deltaR = 0.035 * p0FlucRecomb;
    }
    int i_step = 0;
    while( (deltaR <=0. || deltaR >= 1.)  && i_step < max_step && mode == 0)
    {
      p0FlucRecomb = curand_uniform(&s)*(p0FlucRecomb_u-p0FlucRecomb_l)+p0FlucRecomb_l;
      deltaR = pars[4] * p0FlucRecomb;
      i_step ++; 
    }


    //random rmean
    float fittingRmean[] = {e_upper,p0Recomb,p1Recomb,p2Recomb};

    
    float delta_rmean = legendreSeries(energy,&fittingRmean[0]);
    float rmean;
    float r_u=0.3; 
    float r_l=-0.2; 
    if(mode == 1) rmean = delta_rmean + pars[3];
    else if (mode == 0) rmean = pars[3] + curand_uniform(&s)*(r_u-r_l) + r_l;
    if(debug)
      rmean =  0.015 * energy + p0Recomb;
    i_step = 0;
    
    while((rmean >= 1. || rmean <= 0.0 ) && i_step < max_step && mode == 0)
    {
      rmean = pars[3] + curand_uniform(&s)*(r_u-r_l) + r_l;
      i_step ++;
    }
    if(rmean >= 1.) rmean = 1.;
    if(rmean <= 0.) rmean = 0.;
    if(deltaR <= 0.) deltaR = 0.;
    if(deltaR >= 1.) deltaR = 1.;



    float effectiveSimuAccept = 1.;//acceptanceCheck
    //let the simulation begins

    const float cutSigma = 1.5;
    // 1) get mean quanta number
    int Nq_mean = (int)(energy / w_eV * 1000.);

    // 2) get actual quanta number by fluctuating quanta number with fano factor
    float Fano = get_fano_factor(simuTypeNR, density, Nq_mean, E_drift);

    int Nq_actual = (int)(curand_normal(&s)*sqrtf(Fano * Nq_mean) + Nq_mean);
    i_step = 0;
    while ( i_step < max_step && (Nq_actual < 0 && Nq_actual > Nq_mean + sqrtf(Fano * Nq_mean) * cutSigma || Nq_actual <  Nq_mean - sqrtf(Fano * Nq_mean) ) && mode == 0 ) 
    {
      Nq_actual = (int)(curand_normal(&s)*sqrtf(Fano * Nq_mean) + Nq_mean);
      i_step++;
    }
    while (i_step < max_step && Nq_actual < 0. && mode == 1)
    {
      Nq_actual = (int)(curand_normal(&s)*sqrtf(Fano * Nq_mean) + Nq_mean);
      i_step++;
    }
    if(Nq_actual <= 0) effectiveSimuAccept = 0.;

    // 3) get quanta number after lindhard factor fluctuation
    int Nq;
    if(mode == 0) Nq = gpu_binomial_truncate(cutSigma, &s, Nq_actual, L);
    else if(mode == 1) Nq = gpu_binomial(&s, Nq_actual, L);


    i_step = 0;
    // 4）get exciton ratio and do fluctuation
    int Ni;
    if(mode == 0)
      Ni = gpu_binomial_truncate(cutSigma, &s, Nq, 1./(1. +NexONi));
    else if(mode == 1)
      Ni = gpu_binomial(&s, Nq, 1./(1. +NexONi));
    int Nex = Nq - Ni;

    // 5) get recomb fraction fluctuation
    float r = curand_normal(&s)*deltaR + rmean;
    i_step=0;
    while ( i_step < max_step && ( r > 1. || r < 0. || r > rmean + cutSigma * deltaR || r < rmean - cutSigma * deltaR ) && mode == 0) 
    {
      r = curand_normal(&s)*deltaR + rmean;
      i_step++;
    }

    while( i_step < max_step && (r > 1. || r < 0.) && mode == 1 )
    {
      r = curand_normal(&s)*deltaR + rmean;
      i_step++;
    }
    //if(r<0.)
    //e
    //  r = 0.;
    //  effectiveSimuAccept = 0.;
    //}
    //if(r > 1.)
    //{
    //  r = 1.;
    //  effectiveSimuAccept = 0.; 
    //}


    // 6) get photon and electron numbers
    int Ne;
    if(mode == 0) Ne= gpu_binomial_truncate(cutSigma, &s, Ni, 1 - r);
    else if (mode == 1) Ne = gpu_binomial(&s, Ni, 1 - r);
    int Nph = Nq - Ne;

    // 7) get drift time 
    float truthz = curand_uniform(&s) * TopDrift;
    float dt = (TopDrift - truthz) / driftvelocity;

    // 8) get g1 hit(phd) number
    //test whether g1 is physical
    float g1 = G1/(1. + P_dphe);
    float g1_true = get_g1_true(dt,g1);
    i_step = 0;
    if(g1_true < 0. || g1_true > 1.) 
      effectiveSimuAccept = 0.;

    int nHitsS1;
    if(mode == 0) nHitsS1= gpu_binomial_truncate(cutSigma, &s, Nph, g1_true);
    else if (mode == 1) nHitsS1= gpu_binomial(&s, Nph, g1_true);
    
    // 9) get s1 in phe #, consider float phe
    int NpheS1;
    if(mode == 0) NpheS1= nHitsS1 + gpu_binomial_truncate(cutSigma, &s, nHitsS1, P_dphe);
    else if (mode == 1) NpheS1 = nHitsS1 + gpu_binomial(&s, nHitsS1, P_dphe);

    // 10) s1 pulse area, with pmt resolution
    float pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    i_step = 0;
    while ( i_step <max_step &&  ( pulseAreaS1 > NpheS1 + cutSigma * sqrtf(NpheS1)*sPEres || pulseAreaS1 < NpheS1 - cutSigma * sqrtf(NpheS1)*sPEres ) && mode == 0 )
    {
      i_step++;
      pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    }

    //while(i_step <max_step && mode == 1)
    //{
    //  i_step++;
    //  pulseAreaS1 = curand_normal(&s)*sqrtf(NpheS1)*sPEres + NpheS1;
    //
    //}

    if(pulseAreaS1 <= 0.)  effectiveSimuAccept = 0.;

    // 11) biased s1 pulse area (same in s2 bias)
    float biasS1, fluctuationS1;
    get_bls(pulseAreaS1, &biasS1, &fluctuationS1);
    float pulseAreaBiasS1 = pulseAreaS1 * (curand_normal(&s)*fluctuationS1 + biasS1);
    i_step = 0;
    while (i_step < max_step && (  pulseAreaBiasS1/pulseAreaS1 > cutSigma * fluctuationS1 +biasS1 || pulseAreaBiasS1/pulseAreaS1 < biasS1 - cutSigma * fluctuationS1) && mode == 0 )
    {
      pulseAreaBiasS1 = pulseAreaS1 * (curand_normal(&s)*fluctuationS1 + biasS1);
      i_step++;
    }

    //while (i_step < max_step && pulseAreaBiasS1 < 0. && mode == 1 )
    //{
    //  pulseAreaBiasS1 = pulseAreaS1 * (curand_normal(&s)*fluctuationS1 + biasS1);
    //  i_step++;
    //}
    if (pulseAreaBiasS1 <= 0.) effectiveSimuAccept = 0.;


    // 12) corrected s1 pulse area
    float InversedS1Correction = get_g1_inverse_correction_factor(dt);
    float pulseAreaS1Cor = pulseAreaBiasS1 * InversedS1Correction;
    if(pulseAreaS1Cor <= 0.)    effectiveSimuAccept = 0.;

    // 13) do electron drifting and extraction
    int Nee;
    if(mode == 0) Nee= gpu_binomial_truncate(cutSigma, &s, Ne, expf(-dt / eLife_us) * eee);
    else if (mode == 1) Nee= gpu_binomial(&s, Ne, expf(-dt / eLife_us) * eee);

    // 14) get s2 hit number 
    float deltaNHitsS2 = curand_normal(&s)*sqrtf(SEGb * (float)Nee);
    //int nHitsS2 = (int) (deltaNHitsS2 + SEGb*(float)Nee);
    double nHitsS2 = (double) (deltaNHitsS2 + SEGb*(float)Nee);
    i_step = 0;
    while (i_step < max_step && ( deltaNHitsS2 > cutSigma * sqrtf(SEGb * (float)Nee) || deltaNHitsS2 < -1. * cutSigma * sqrtf(SEGb * (float)Nee) ) && mode == 0)
    {
      i_step++;
      deltaNHitsS2 = curand_normal(&s)*sqrtf(SEGb * (float)Nee);
      //nHitsS2 = (int) (deltaNHitsS2 + SEGb*(float)Nee);
      nHitsS2 = (double) (deltaNHitsS2 + SEGb*(float)Nee);
    }

    //while( i_step < max_step && nHitsS2 < 0 && mode == 1)
    //{
    //  i_step++;
    //  deltaNHitsS2 = curand_normal(&s)*sqrtf(SEGb * (float)Nee);
    //  nHitsS2 = (int) (deltaNHitsS2 + SEGb*(float)Nee);
    //}

    if(nHitsS2 <= 0.)    effectiveSimuAccept = 0.;

    // 15) get s2 phe
    int NpheS2;
    if(mode == 0) NpheS2 = nHitsS2 + gpu_binomial_truncate(cutSigma, &s, int(nHitsS2), P_dphe);
    else if (mode == 1) NpheS2 = nHitsS2 + gpu_binomial(&s, int(nHitsS2), P_dphe);

    // 16) s2 pulse area, with pmt resolution
    i_step = 0;
    float pulseAreaS2 = curand_normal(&s)*sqrtf(NpheS2)*sPEres + NpheS2; 
    while (i_step < max_step && (  pulseAreaS2 > cutSigma * sqrtf(NpheS2)*sPEres + NpheS2 ||pulseAreaS2 < NpheS2 - cutSigma * sqrtf(NpheS2)*sPEres ) && mode == 0) 
    {
      i_step++;
      pulseAreaS2 = curand_normal(&s)*sqrtf(NpheS2)*sPEres + NpheS2;
    }
    //while (i_step < max_step && pulseAreaS2 < 0 && mode == 1)
    //{
    //  i_step++;
    //  pulseAreaS2 = curand_normal(&s)*sqrtf(NpheS2)*sPEres + NpheS2;
    //}

    if(pulseAreaS2 <= 0.)  effectiveSimuAccept = 0.;
    

    // 17) biased s2 pulse area 
    float biasS2, fluctuationS2;
    get_bls(pulseAreaS2, &biasS2, &fluctuationS2);
    float pulseAreaBiasS2 = pulseAreaS2 * (curand_normal(&s)*fluctuationS2 + biasS2);
    i_step = 0;
    while( i_step < max_step &&  ( pulseAreaBiasS2/pulseAreaS2 > biasS2 + cutSigma * fluctuationS2 || pulseAreaBiasS2/pulseAreaS2 < biasS2 - cutSigma * fluctuationS2 ) && mode == 0 )
    {
      pulseAreaBiasS2 = pulseAreaS2 * (curand_normal(&s)*fluctuationS2 + biasS2);
      i_step++;
    }

    //while(i_step < max_step && pulseAreaBiasS2 < 0. && mode == 1)
    //{
    //  pulseAreaBiasS2 = pulseAreaS2 * (curand_normal(&s)*fluctuationS2 + biasS2);
    //  i_step++;
    //}
    if(pulseAreaBiasS2 <= 0.)  effectiveSimuAccept = 0.;

    // 18）corrected s2 pulse area
    float InversedS2Correction = get_g2_inverse_correction_factor(dt, eLife_us);
    float pulseAreaS2Cor = pulseAreaBiasS2 * InversedS2Correction;    
    //float pulseAreaS2CorB = gpu_binomial_truncate(cutSigma, &s, int(pulseAreaS2Cor), bottom_fraction);
    if(pulseAreaS2Cor <= 0.)
    {
      effectiveSimuAccept = 0.;
    }
    float pulseAreaS2CorB = pulseAreaS2Cor;

    if(pulseAreaS1Cor <= 0.||pulseAreaS2Cor <= 0.) effectiveSimuAccept = 0.;
    
    float s1s2RangeAccept = 1.;
    if ( int(nHitsS1) <= 1 || pulseAreaS2CorB < 20. || pulseAreaS1Cor < 2. )
    {
      s1s2RangeAccept = 0.;
    }
    
    float par_e[]={flatE}; 
    float weight = get_energy_weight(energy,par_e,energyType);
    if(weight<=0.)weight = 0.;

    float TotalAcceptance = float(effectiveSimuAccept) * float(s1s2RangeAccept);
    //float TotalAcceptance = float(s1s2RangeAccept);

    float eff = 1.;
    //s1_eff for tritium
    float par_s1_eff[]={float(const_pars[4]),float(const_pars[5])};
    eff *= get_efficiency_FermiDirac(pulseAreaS1Cor,&par_s1_eff[0]);

    float par_s1tag_eff[]={float(const_pars[6]),float(const_pars[7])};
    eff *= get_efficiency_FermiDirac(pulseAreaS1Cor,&par_s1tag_eff[0]);

    float par_s2b_eff[]={float(const_pars[8]),float(const_pars[9])};
    eff *= get_efficiency_FermiDirac(pulseAreaS2CorB,&par_s2b_eff[0]);
    if(pulseAreaS2CorB<0. || pulseAreaS1Cor<0. || isnan(pulseAreaS1Cor)||isnan(pulseAreaS2CorB))
    {
      eff = 0.;
      TotalAcceptance = 0.;
    }

    float eff_weight = weight * TotalAcceptance * eff; 
    //if( (isnan(weight)||isinf(weight) ) && iteration <100000) printf("%d, %f, %f, %f, %f\\n",iteration,pulseAreaS2CorB,pulseAreaS1Cor,weight,energy);

    //printf("This is the %d-th GPU thread! %.2f\\n", iteration, pulseAreaS2CorB);
    //similar to branch fill in ROOT
    float* iter_data = (float*)((char*)output_data + iteration * stride_output_data);
    *(iter_data)=(float)pulseAreaS1Cor;
    *(iter_data+1)=(float)pulseAreaS2Cor;
    *(iter_data+2)=(float)dt;
    *(iter_data+3)=(float)eLife_us;
    *(iter_data+4)=(float)G1;
    *(iter_data+5)=(float)Nph;
    *(iter_data+6)=(float)nHitsS1;
    *(iter_data+7)=(float)eee;
    *(iter_data+8)=(float)Ne;
    *(iter_data+9)=(float)Nee;
    *(iter_data+10)=(float)eff_weight;
    *(iter_data+11)=(float)SEGb;
    *(iter_data+12)=(float)nHitsS2;
    //*(iter_data+13)=(float)bottom_fraction;
    *(iter_data+14)=(float)pulseAreaS2CorB;
    *(iter_data+15)=(float)pulseAreaBiasS2;
    *(iter_data+16)=(float)L;
    *(iter_data+17)=(float)r;
    *(iter_data+18)=(float)rmean;
    *(iter_data+19)=(float)deltaR;
    *(iter_data+20)=(float)energy;
    *(iter_data+21)=(float)Nq_actual;
    *(iter_data+22)=(float)Nq;
    *(iter_data+23)=(float)TotalAcceptance;
    *(iter_data+24)=(float)weight;
    
    if(debug)
    {
      *(iter_data+27)=(float)p0Recomb;
      *(iter_data+24)=(float)p1Recomb;
      *(iter_data+25)=(float)p2Recomb;
      *(iter_data+26)=(float)p0FlucRecomb;

    }

    //if(weight > 10 &&iteration<1000)
    if(iteration<10 )
    {
      printf("check %d :  %.2f %.2f %.2f %.2f %.2f %.2f\\n",iteration, r,float(Nph),float(Ne),float(nHitsS1), float(nHitsS2),rmean);
    }


  }

  __global__ void fill_histogram(float* simu_tree, int stride_simu_tree,int N_simu_tree,
    double *output, double *hist2d_par)
  {
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;


    // keep only the GPU nodes that satisfy the number of trial
    if (iteration>N_simu_tree) return;

    float* iter_data = (float*)((char*)simu_tree + iteration * stride_simu_tree);
    float pulseAreaS1Cor = *(iter_data);
    float pulseAreaS2CorB = *(iter_data+14);
    float ori_weight = *(iter_data+24);
    float eff_weight = *(iter_data+10);

    //filling ouput 2d histogram
    int xBinning = (int)*(hist2d_par+1);
    double xMin = (double)*(hist2d_par+2);
    double xMax = (double)*(hist2d_par+3);
    double xStep = (xMax - xMin)/(double)xBinning;
    int yBinning = (int)*(hist2d_par+4);
    double yMin = (double)*(hist2d_par+5);
    double yMax = (double)*(hist2d_par+6);
    double yStep = (yMax - yMin)/(double)yBinning;

    //get values, overflows and underflows in output[1]-output[9]
    double xvalue = (double)pulseAreaS1Cor;
    double yvalue = (double)log10f(pulseAreaS2CorB);

    //printf("I am block%d and thread%d for iteration%d \\n",blockIdx.x, threadIdx.x, iteration);
    if(ori_weight > 0.0)
    {
      atomicAdd(hist2d_par+7,ori_weight);
      if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
      {
        int xbin = floorf((xvalue - xMin)/xStep) ;
        int ybin = floorf((yvalue - yMin)/yStep) ;
        double* data_address = (double*) ((char*)output + ybin*xBinning * sizeof(double)) + xbin;
        atomicAdd(data_address,eff_weight);
        atomicAdd(hist2d_par,eff_weight);
      }

    }

  }


}

"""
