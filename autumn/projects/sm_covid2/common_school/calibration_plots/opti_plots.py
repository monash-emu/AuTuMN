from matplotlib import pyplot as plt 
from math import ceil
import os

def flatten_p_dict(p_dict):
    flat_dict = {p: val for p, val in p_dict.items() if p != 'random_process.delta_values'}
    rpdv = p_dict['random_process.delta_values']
    for i, dv in enumerate(rpdv):
        flat_dict[f"random_process.delta_values[{i}]"] = dv

    return flat_dict


def plot_opti_params(sample_as_dicts, best_params, bcm, output_folder):
    flat_best_params = {i: flatten_p_dict(p_dict) for i, p_dict in best_params.items()}

    n_params = len(flat_best_params[0])
    n_samples = len(sample_as_dicts)
    n_cols = 3
    n_rows = ceil(n_params / n_cols) 

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4. * n_cols, 1.3 * n_rows))
    cmap = plt.cm.get_cmap("plasma", n_samples)
    colors = [cmap(i) for i in range(n_samples)]

    i_row, i_col = 0 , 0
    for i, p_name in enumerate(flat_best_params[0]):
        ax = axs[i_row, i_col]
        
        if p_name.startswith("random_process.delta_values"):
            bounds = (-1., 1.)
            start_vals = [0.] * n_samples
        elif p_name in sample_as_dicts[0]:
            bounds = bcm.priors[p_name].bounds()
            start_vals = [sample_as_dicts[i][p_name] for i in range(n_samples)]
        else:
            bounds = bcm.priors[p_name].bounds()
            start_vals = [(bounds[0] + bounds[1]) / 2.] * n_samples

        ax.plot(bounds, (0,0), color="black", lw=.5, zorder=-10)       
        p_vals = [flat_best_params[i][p_name] for i in flat_best_params]
        ax.scatter(x=p_vals, y=[0.] * len(p_vals), c=colors)
        
        ax.plot(bounds, (1., 1.), color="black", lw=.5, zorder=-10)       
        ax.scatter(x=start_vals, y=[1.] * len(p_vals), c = colors)
        if not p_name.startswith("random_process.delta_values"):
            for k in range(n_samples):
                ax.text(x=start_vals[k], y=.7, s=str(k), size=6, ha="center")

        for i in range(len(start_vals)):
            ax.plot([start_vals[i], p_vals[i]], [1., 0.], lw=.3, color=colors[i])

        ax.get_yaxis().set_visible(False)
        
        ax.set_xticks(bounds, bounds, fontsize=8)
        # ax.get_xaxis().set_ticks_position("none")

        ax.set_ylim(-.4, 1.6)

        ax.text(bounds[0], 0.1, "opti", size=7.)
        ax.text(bounds[0], 1.1, "init", size=7.)

        ax.text((bounds[0] + bounds[1]) / 2., 1.2, p_name, size=8., ha="center")
        i_row += 1
        if i_row == n_rows:
            i_row = 0
            i_col += 1

    for remain_i_row in range(i_row, n_rows):
        ax = axs[remain_i_row, i_col]
        ax.set_visible(False)

    fig.tight_layout()
   
    fig.savefig(os.path.join(output_folder, "opti_params.png"), facecolor="white")

    plt.close()


def plot_model_fit(bcm, params, outfile=None):
    REF_DATE = datetime.date(2019,12,31)

    targets = {}
    for output in bcm.targets:
        t = copy(bcm.targets[output].data)
        t.index = ref_times_to_dti(REF_DATE, t.index)
        targets[output] = t

    run_model = bcm.run(params)
    ll = bcm.loglikelihood(**params)  # not ideal...

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), height_ratios=(2., 1., 2.), sharex=True)
    death_ax, rp_ax, sero_ax = axs[0], axs[1], axs[2]

    # Deaths
    run_model.derived_outputs["infection_deaths"].plot(ax=death_ax, ylabel="COVID-19 deaths")
    targets["infection_deaths"].plot(style='.', ax=death_ax)
    plt.text(0.8, 0.9, f"ll={round(ll, 4)}", transform=death_ax.transAxes)

    # Random Process
    run_model.derived_outputs["transformed_random_process"].plot(ax=rp_ax, ylabel="Random Process")
    y_max = max(rp_ax.get_ylim()[1], 1.1)
    xmin, xmax = rp_ax.get_xlim()
    rp_ax.set_ylim(0, y_max)
    rp_ax.hlines(y=1., xmin=xmin, xmax=xmax, linestyle="dotted", color="grey")

    # Sero data
    if "prop_ever_infected_age_matched" in targets:
        run_model.derived_outputs["prop_ever_infected_age_matched"].plot(ax=sero_ax, ylabel="Prop. ever infected\n(age-matched)")
        targets["prop_ever_infected_age_matched"].plot(style='.', ax=sero_ax)
    else:
        run_model.derived_outputs["prop_ever_infected"].plot(ax=sero_ax, ylabel="Prop. ever infected")

    if outfile:
        fig.savefig(outfile, facecolor="white")

    return fig