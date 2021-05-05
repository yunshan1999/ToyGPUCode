#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>

{
const int xbinning = 100;
const int ybinning = 100;
double xmin = 2.;
double xmax = 120.;
double ymin = 0.;
double ymax = log10(20000.);
double xstep = (xmax-xmin)/(double)xbinning;
double ystep = (ymax-ymin)/(double)ybinning;
TH2D * h1 = new TH2D("mc", "", xbinning, xmin, xmax, ybinning, ymin, ymax);
std::ifstream infile;
infile.open("../FittingGPU/outputs/2dband.dat");
const int N = 11 + xbinning*ybinning;
double encoded_array[N] = {0};
if (infile.is_open()) 
{
	for(int i = 0 ;i < N; i++)
	{
		infile>>encoded_array[i];

	}
}
else
{
	std::cout<<"error in file opening"<<std::endl;
}
for(int i = 11; i < N; i++)
{
        int ybin = (i-10)/xbinning + 1;
        int xbin = (i-10)-xbinning*(ybin-1);
        if(xbin==0)
        {
            ybin = (i-10)/xbinning;
            xbin = xbinning;
        }
        h1->SetBinContent(xbin, ybin, encoded_array[i]);
}
//get quantiles of band//

double x[xbinning], y05[xbinning], y50[xbinning], y95[xbinning];
for(int i = 0; i < xbinning; i++)
{
        int firstbin = i+1;
        int lastbin = i+1;
        double aprob[3] = {0.05,0.5,0.95};
        double median[3];

        TH1D * h_temp = h1->ProjectionY("proj",firstbin,lastbin);
        if(h_temp->Integral()==0.)continue;
        h_temp->GetQuantiles(3,median,aprob);
        x[i] =  xstep*i + xmin;
        y05[i] = median[0];
        y50[i] = median[1];
        y95[i] = median[2];
}

TCanvas * c1 = new TCanvas();
c1->SetTitle("MC band");
h1->SetStats(0);
h1->SetXTitle("cS1[PE]");
h1->SetYTitle("log(cS2/cS1)");
h1->Draw("colz");
c1->Print( "../FittingGPU/outputs/MCband.pdf)","pdf");

TCanvas * c2 = new TCanvas();
TFile dataFile("../FittingGPU/data/reduce_ana3_p4_run1_tritium_5kV.root","read");
TTree * data_tree=(TTree*) dataFile.Get("out_tree");
data_tree->Draw("log10(qS2BC_max):qS1C_max>>h2(100,xmin,xmax, 100, ymin, ymax)","qS1C_max>2&&qS1C_max<120","");

TGraph * g1 = new TGraph(xbinning, x, y05);
TGraph * g2 = new TGraph(xbinning, x, y50);
TGraph * g3 = new TGraph(xbinning, x, y95);
g1->SetLineColor(2);
g1->SetLineStyle(2);
g1->SetLineWidth(1);
g2->SetLineColor(2);
g2->SetLineStyle(2);
g2->SetLineWidth(1);
g3->SetLineColor(2);
g3->SetLineStyle(2);
g3->SetLineWidth(1);

g1->Draw("SAME");
g2->Draw("SAME");
g3->Draw("SAME");
c2->Print( "../FittingGPU/outputs/DATAband.pdf)","pdf");
}
