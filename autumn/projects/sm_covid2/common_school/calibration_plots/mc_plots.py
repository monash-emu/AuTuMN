from matplotlib import pyplot as plt
from pathlib import Path
import arviz as az


def make_post_mc_plots(idata, burn_in, output_folder=None):
    az.rcParams["plot.max_subplots"] = 60 # to make sure all parameters are included in trace plots

    if output_folder:
        output_folder_path = Path(output_folder) / "mc_outputs"
        output_folder_path.mkdir(exist_ok=True, parents=True)

    chain_length = idata.sample_stats.sizes['draw']

    # Traces (including burn-in)
    az.plot_trace(idata, figsize=(16, 5.0 * len(idata.posterior)), compact=False);
    plt.subplots_adjust(hspace=.7)
    if output_folder:
        plt.savefig(output_folder_path / "mc_traces.jpg", facecolor="white", bbox_inches='tight')
        plt.close()

    # burn data
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in

    # Traces (after burn-in)
    az.plot_trace(burnt_idata, figsize=(16, 5.0 * len(idata.posterior)), compact=False);
    plt.subplots_adjust(hspace=.7)
    if output_folder:
        plt.savefig(output_folder_path / "mc_traces_postburnin.jpg", facecolor="white", bbox_inches='tight')
        plt.close()

    # Posteriors (excluding burn-in)
    az.plot_posterior(burnt_idata);
    if output_folder:
        plt.savefig(output_folder_path / "mc_posteriors_postburnin.png", facecolor="white", bbox_inches='tight')
        plt.close()

    # ESS (excluding burn-in)
    raw_ess_df = az.ess(burnt_idata).to_dataframe()
    ess_df = raw_ess_df.drop(columns="random_process.delta_values").loc[0]
    for i in range(len(raw_ess_df)):
        ess_df[f"random_process.delta_values[{i}]"] = raw_ess_df['random_process.delta_values'][i]
    if output_folder:
        ess_df.to_csv(output_folder_path / "mc_ess.csv", header=["ESS"])

    # R_hat plot (excluding burn-in)
    raw_rhat_df = az.rhat(burnt_idata).to_dataframe()
    rhat_df = raw_rhat_df.drop(columns="random_process.delta_values").loc[0]
    for i in range(len(raw_rhat_df)):
        rhat_df[f"random_process.delta_values[{i}]"] = raw_rhat_df['random_process.delta_values'][i]
    axis = rhat_df.plot.barh(xlim=(1.,1.105))
    axis.vlines(x=1.05,ymin=-0.5, ymax=len(rhat_df), linestyles="--", color='orange')
    axis.vlines(x=1.1,ymin=-0.5, ymax=len(rhat_df), linestyles="-",color='red')    
    if output_folder:
        plt.savefig(output_folder_path / "r_hats.jpg", facecolor="white", bbox_inches='tight')
        plt.close()
