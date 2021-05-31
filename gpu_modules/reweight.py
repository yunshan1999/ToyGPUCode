pandax4t_reweight_fitting = """
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
    return 0.;

  }


  __device__ float get_efficiency_FermiDirac(float xx,float *par){
    float eff = 1./(expf(-1.*(xx-par[0])/par[1]) + 1.);
    return eff;
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


  __device__ float truncated_normal_ratio(float obs, float mean1, float sigma1, float mean2, float sigma2, float lower, float upper)
  {
    if(sigma1 == 0. || sigma2 == 0.)
    {
      if(sigma1 == sigma2 && mean1 == mean2) return 1.;
      else return 0.;
    }

    float R = 1.;
    if (obs<lower || obs>upper) return 0;
    // need to calculate the scale factors for trucated gaussian
    float scale_factor1 = 0.5*(erff( (upper-mean1)/sigma1 / sqrtf(2.)) - erff( (lower-mean1)/sigma1/sqrtf(2.) ));
    float scale_factor2 = 0.5*(erff( (upper-mean2)/sigma2 / sqrtf(2.)) - erff( (lower-mean2)/sigma2/sqrtf(2.) ));
    R *= scale_factor2/scale_factor1 * sigma2 / sigma1;//denominator terms
    R *= expf( - 0.5*powf((obs - mean1)/sigma1, 2.) + 0.5*powf( (obs-mean2)/sigma2, 2.) );//numerator terms
    return R;
  }


  __device__ float normal_ratio(float obs, float mean1, float sigma1, float mean2, float sigma2)
  {
    if(sigma1 == 0. || sigma2 == 0.)
    {
      if(sigma1 == sigma2 && mean1 == mean2) return 1.;
      else return 0.;
    }
    double R = 1.;
    R *= double(sigma2) / double(sigma1);
    R *= exp( - 0.5*pow((double(obs) - double(mean1))/double(sigma1), 2.) + 0.5*pow( (double(obs)-double(mean2))/double(sigma2), 2.) );
    return float(R);
  }


  
  __device__ float binomial_ratio(int obs, int N, float prob1, float prob2)
  {
    if(N<0 || obs <0) return 0.;
    if( float(prob1) == float(prob2) ) return 1.;
    else if(prob2 == 1. || prob2 == 0. || prob1 == 1. || prob1 == 0.) return 0.;

    if(float(N) * prob2 > 20 && float(N) * (1. - prob2) > 20){
      float sigma_new = sqrtf(float(N) * prob1 * (1. - prob1));
      float mean_new  = float(N) * prob1;
      float sigma = sqrtf(float(N) * prob2 * (1. - prob2));
      float mean = float(N) * prob2;
      return normal_ratio(float(obs), mean_new,sigma_new,mean,sigma);
    
    }

    float R = 1.;
    R *= pow(double(prob1)/double(prob2), (double)obs);
    R *= pow((1.-double(prob1))/(1.-double(prob2)), (double)(N - obs));
    return R;
  }


  __global__ void ln_likelihood(
      float* xdata, float* ydata, 
      double *histogram2d, double *histogram2d_par,
      double *lnlikelihood, int Ndata)
  {
    int eventId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; //start with 1 
    //printf("test i: %d %d",int(*Ndata),eventId);

    if(eventId > (Ndata) ) return;
    float s1 = (float) *(xdata+eventId);
    float s2 = (float) log10(double(*(ydata+eventId)));


    double Nmc = (double) *(histogram2d_par);
    int xBinning = (int)*(histogram2d_par+1);
    float xMin = (float)*(histogram2d_par+2);
    float xMax = (float)*(histogram2d_par+3);
    float xStep = (double(xMax) - double(xMin))/double(xBinning);
    int yBinning = (int)*(histogram2d_par+4);
    float yMin = (float)*(histogram2d_par+5);
    float yMax = (float)*(histogram2d_par+6);
    float yStep = (double(yMax) - double(yMin))/double(yBinning);

    int xbin = (int) floor((double(s1) - double(xMin))/double(xStep)) ;
    int ybin = (int) floor((double(s2) - double(yMin))/double(yStep)) ;
    double bincontent = *((double*) ((char*)histogram2d + ybin*xBinning * sizeof(double)) + xbin);
    //if(eventId <20) printf("test s1, log10(s2),bincontent: %.2f, %.2f, %.2f\\n",s1,s2,bincontent);
    //double bincontent = *data_address;
    if (bincontent == 0.0) atomicAdd(&lnlikelihood[0],-100.0);
    else if (bincontent > 0.0) atomicAdd(&lnlikelihood[0],(log10(bincontent/Nmc)));


  }



  __device__ void get_yield_pars(bool simuTypeNR, float E_drift, float energy, float density, float * pars){
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
      pars[3] = rmean ;
      pars[4] = omega ;
      //if(pars[3]<0.)pars[3] = 0.;
      //if(pars[3]>0.9)pars[3] = 0.9;
      //if(pars[4]<0.001)pars[4] = 0.001;
      return;
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
      pars[4] = omega ;
      //if(pars[3]<0.)pars[3] = 0.;
      //if(pars[3]>0.9)pars[3] = 0.9;
      //if(pars[4]<0.)pars[4] = 0.;
      return;
    }
  }

  __global__ void leakage_ratio_calculation(
  float *e_reweight_x,float *e_reweight_y,
  float *inTree, float *inTree_par,
  float *out_weight,float *out_normalization,
  double *output_par)
  {

    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // get the number of trials
    int num_trials      = (int) *(inTree_par);
    if (iteration>num_trials) return;
    int stride_inTree = (int) *(inTree_par+1);
    float* entry_data = (float*)((char*)inTree + iteration * stride_inTree);
    float pulseAreaS1Cor = (float) *(entry_data);
    float pulseAreaS2CorB = (float) *(entry_data + 14);
    float energy = (float) *(entry_data+20);

    float energy_dRdE = (float) *(out_normalization + iteration);

    float weight =  (float) *(out_weight + iteration);
    float filledEnergyWeight=interpolate1d(energy,e_reweight_x,e_reweight_y);

    float renormalizationW = 0.;
    if(weight >= 0. && filledEnergyWeight > 0. && energy_dRdE > 0.) renormalizationW= weight * energy_dRdE / filledEnergyWeight;

    *(out_weight + iteration) = (float) renormalizationW;
    double check = 1.2328+0.497802*exp(-pulseAreaS1Cor/8.25663)+(-0.00422753*pulseAreaS1Cor);
    if(pulseAreaS1Cor < 45 && renormalizationW > 0)
    {
      if(log10(pulseAreaS2CorB/pulseAreaS1Cor)<check)
        atomicAdd(output_par,renormalizationW);
      atomicAdd(output_par+1,renormalizationW);
    }
  }



  __global__ void renormalization_reweight(
  float *e_reweight_x,float *e_reweight_y,
  float *inTree, float *inTree_par, 
  float *out_weight,float *out_normalization,float *out_eff,
  double *output,double *output_par)
  {
     
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // get the number of trials
    int num_trials      = (int) *(inTree_par);
    if (iteration>num_trials) return;
    int stride_inTree = (int) *(inTree_par+1);
    float* entry_data = (float*)((char*)inTree + iteration * stride_inTree);
    float pulseAreaS1Cor = (float) *(entry_data);
    float pulseAreaS2CorB = (float) *(entry_data + 14);
    float energy = (float) *(entry_data+20);
    
    float energy_dRdE = (float) *(out_normalization + iteration);
    float eff = (float) *(out_eff + iteration);

    float weight =  (float) *(out_weight + iteration);
    float filledEnergyWeight=interpolate1d(energy,e_reweight_x,e_reweight_y);
    
    float renormalizationW = 0.;
    float tot_renormalizationW = 0.;
    if(weight >= 0. && filledEnergyWeight > 0. && energy_dRdE > 0.) 
    {
      renormalizationW= weight * energy_dRdE * eff / filledEnergyWeight;
      tot_renormalizationW = weight * energy_dRdE /filledEnergyWeight; 
    }
    if(isnan(renormalizationW)) printf("check weight %f %f %f \\n",weight,energy_dRdE,filledEnergyWeight);

    *(out_weight + iteration) = (float) renormalizationW;
   
   //filling ouput 2d histogram
    int xBinning = (int)*(output_par+1);
    double xMin = (double)*(output_par+2);
    double xMax = (double)*(output_par+3);
    double xStep = (xMax - xMin)/(double)xBinning;
    int yBinning = (int)*(output_par+4);
    double yMin = (double)*(output_par+5);
    double yMax = (double)*(output_par+6);
    double yStep = (yMax - yMin)/(double)yBinning;

    //get values, overflows and underflows in output[1]-output[9]
    double xvalue = (double)pulseAreaS1Cor;
    double yvalue = (double)log10f(pulseAreaS2CorB);

    //printf("I am block%d and thread%d for iteration%d \\n",blockIdx.x, threadIdx.x, iteration);
    if(tot_renormalizationW > 0.0)
    {
      atomicAdd(output_par+7,tot_renormalizationW);
      if(xvalue>=xMin && xvalue<xMax && yvalue>=yMin && yvalue<yMax)
      {
        int xbin = floorf((xvalue - xMin)/xStep) ;
        int ybin = floorf((yvalue - yMin)/yStep) ;
        double* data_address = (double*) ((char*)output + ybin*xBinning * sizeof(double)) + xbin;
        atomicAdd(data_address,renormalizationW);
        atomicAdd(output_par,renormalizationW);

      }
    }

  }





  __global__ void main_reweight_fitting(float *new_v_variable,
      float *inTree, float *inTree_par,
      float *const_pars, double *output_par,
      float *out_reweight, float *out_normalization,float *out_eff,
      int mode)
  {
    // initiate the random sampler, must have.
    int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // get the number of trials


    int num_trials      = (int) *(inTree_par);
    int stride_inTree = (int) *(inTree_par+1);
    
    float stepE = (float) const_pars[0];
    float minE = (float) const_pars[2];
    float maxE = (float) const_pars[3];

    if (iteration>num_trials) return;

    float* entry_data = (float*)((char*)inTree + iteration * stride_inTree);

    float pulseAreaS1Cor = (float) *(entry_data);
    float pulseAreaS2Cor = (float) *(entry_data+1);
    float dt = (float) *(entry_data+2);
    float eLife_us = (float) *(entry_data+3);

    float G1 = (float) *(entry_data+4);
    float Nph = (float) *(entry_data+5);
    float nHitsS1 = (float) *(entry_data+6);

    float eee = (float) *(entry_data+7);
    float Ne = (float) *(entry_data+8);
    float Nee = (float) *(entry_data+9);
    float weight = (float) *(entry_data+10);
    float seg = (float) *(entry_data+11);
    float nHitsS2 = (float) *(entry_data+12);
    //float bottom_fraction = (float) *(entry_data+13);
    float pulseAreaS2CorB = (float) *(entry_data+14);
    float pulseAreaBiasS2 = (float) *(entry_data+15);
    float L = (float) *(entry_data+16);
    float r = (float) *(entry_data+17);
    float rmean = (float) *(entry_data+18);
    float deltaR = (float) *(entry_data+19);
    float energy = (float) *(entry_data+20);
    float Nq_actual = (float) *(entry_data+21);
    float Nq = (float) *(entry_data+22);
    float TotalAcceptance = (float) *(entry_data+23);



    //the real random walk space is half of the initial data for every dimension
    float G1_new =  (float) *(new_v_variable);
    float g2b_new =  (float) *(new_v_variable+1);
    float seg_new =  (float) *(new_v_variable+2);
    float p2Recomb_new =  (float) *(new_v_variable+3);
    float p0Recomb_new = (float) *(new_v_variable+4);
    float p1Recomb_new = (float) *(new_v_variable+5);
    float p0FlucRecomb_new = (float) *(new_v_variable+6);
    float flatE_new = (float) *(new_v_variable+7);
    
    bool simuTypeNR = bool(const_pars[1]);
    int energyType = int(const_pars[10]);
    float E_drift = const_pars[13]; //drift electric field in V/cm
    float P_dphe = const_pars[12]; //probab that 1 phd makes 2nd phe
    float density = 2.8611; //density of liquid xenon

    //if(iteration<2)
    //{
    //  printf("energy: %d %d %f %f %f\\n",num_trials,stride_inTree,E_drift,P_dphe,flatE_new);
    //}
    float g1_new = G1_new / (1. + P_dphe);
    float eee_new = g2b_new / seg_new;
    float par_e[]={flatE_new};
    float energy_weight_new = get_energy_weight(energy,par_e,energyType);
    if(energy_weight_new < 0. || isnan(energy_weight_new))energy_weight_new =0.;
    
    float pars[5]={0};


    get_yield_pars(simuTypeNR, E_drift, energy, density, &pars[0]);
    float L_new = pars[1];//lindhard factor
    float fittingRmean[] = {maxE,p0Recomb_new,p1Recomb_new,p2Recomb_new};
    float rmean_new = pars[3] + legendreSeries(energy,&fittingRmean[0]);
    float deltaR_new = pars[4] * p0FlucRecomb_new;

    bool debug = false;
    if(debug)
    {
      //rmean_new  = pars[3] + legendreSeries(energy,&fittingRmean[0]);
      rmean_new  = 0.015 * energy + legendreSeries(energy,&fittingRmean[0]);
      deltaR_new = 0.035 * p0FlucRecomb_new;
    }

    float g1 = G1 / (1. + P_dphe);
    float g1_true = get_g1_true(dt,g1);
    float g1_true_new = get_g1_true(dt,g1_new);

    double ReweightingWeight = 1.;
    ReweightingWeight *= (double) binomial_ratio(Nq, Nq_actual, L_new,L);
    double r1 = (double) truncated_normal_ratio(r,rmean_new,deltaR_new,rmean,deltaR,0.,1.);
    //double r1 = (double) normal_ratio(r,rmean_new,deltaR_new,rmean,deltaR);
    //double r2 = (double) binomial_ratio(int(pulseAreaS2CorB),int(pulseAreaS2Cor),bottom_fraction_new,bottom_fraction);
    double r3 = (double) binomial_ratio(int(nHitsS1), int(Nph), g1_true_new, g1_true);
    double r4 = (double) binomial_ratio(int(Nee), int(Ne), expf(-dt / eLife_us) * eee_new,expf(-dt / eLife_us) * eee);
    double r5 = (double) normal_ratio(nHitsS2, double(seg_new) * double(Nee), sqrt(double(seg_new) * double(Nee)), double(seg) * double(Nee), sqrt(double(seg) * double(Nee)));
    ReweightingWeight *= double(r1 *  r3 * r4 * r5);
    //ReweightingWeight *= r1 ;

    if(isnan(ReweightingWeight) || isinf(ReweightingWeight)) 
    {
      ReweightingWeight = 0.;
      //printf("test weight %f %f %f %f",r1, r3, r4, r5);
      //printf("%d: r, energy, rmean, deltaR,rmean_new,deltaR_new,r1: %f %f %f %f %f %f %f\\n",iteration,r,energy,rmean,deltaR,rmean_new,deltaR_new,r1);
    }
//    if (iteration < int(10000) && r < 0.15 && energy < 1.4 && ReweightingWeight > 0.0)
//    {
//        printf("ori: g1, eee, seg, reweighting: %.2f, %.2f, %.2f, %f %f %f %f %f %f\\n",g1,eee,seg,ReweightingWeight, r1,r3,r4,r5,double(r1*r3*r4*r5));
//    }
//
    


    float eff = 1.;
    
    //s1_eff for tritium
    float par_s1_eff[]={float(const_pars[4]),float(const_pars[5])};
    eff *= get_efficiency_FermiDirac(pulseAreaS1Cor,&par_s1_eff[0]);
    
    float par_s1tag_eff[]={float(const_pars[6]),float(const_pars[7])};
    eff *= get_efficiency_FermiDirac(pulseAreaS1Cor,&par_s1tag_eff[0]);

    float par_s2b_eff[]={float(const_pars[8]),float(const_pars[9])};
    eff *= get_efficiency_FermiDirac(pulseAreaS2CorB,&par_s2b_eff[0]);
    if(isnan(eff)||isinf(eff)) eff = 0.;


    if(mode == 1 )
    {

      *(out_reweight + iteration ) = (float)  ReweightingWeight;
      *(out_normalization+iteration) =  (float) energy_weight_new; 
      *(out_eff+iteration) =  (float)  TotalAcceptance * eff; 
      float test = (float) energy_weight_new * TotalAcceptance * eff;
      if(isnan(test) )printf("check %d %f %f %f %f",iteration,energy_weight_new,TotalAcceptance,ReweightingWeight,eff);
      //*(out_normalization+iteration) =  (float) energy_weight_new * eff;
      //fill reweight energy spec
      int offSetE = 7;
      if (energy>=minE && energy <=maxE ) 
      {
        int binOfE = floorf((energy - minE)/stepE);
        double* e_address = (double*) ((char*)output_par + ( offSetE + binOfE) * sizeof(double));
        atomicAdd(e_address, ReweightingWeight); 
        atomicAdd(output_par, ReweightingWeight);
      }
      //if (iteration == num_trials )printf("catch %f",ReweightingWeight);
    }



  }




}
"""
