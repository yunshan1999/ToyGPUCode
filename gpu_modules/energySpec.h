__device__ get_tritium_energy_weight(float energy,float *par)
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
