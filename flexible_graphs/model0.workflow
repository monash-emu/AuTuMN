digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbidities
		active_smearpos_ds_nocomorbidities
		active_smearneg_ds_nocomorbidities
		active_extrapul_ds_nocomorbidities
		susceptible_vac_nocomorbidities
		susceptible_treated_nocomorbidities
		latent_early_ds_nocomorbidities
		latent_late_ds_nocomorbidities
		detect_smearpos_ds_nocomorbidities
		detect_smearneg_ds_nocomorbidities
		detect_extrapul_ds_nocomorbidities
		missed_smearpos_ds_nocomorbidities
		missed_smearneg_ds_nocomorbidities
		missed_extrapul_ds_nocomorbidities
		treatment_infect_smearpos_ds_nocomorbidities
		treatment_infect_smearneg_ds_nocomorbidities
		treatment_infect_extrapul_ds_nocomorbidities
		treatment_noninfect_smearpos_ds_nocomorbidities
		treatment_noninfect_smearneg_ds_nocomorbidities
		treatment_noninfect_extrapul_ds_nocomorbidities
		tb_death
			susceptible_fully_nocomorbidities -> latent_early_ds_nocomorbidities [label=rate_force_ds]
			susceptible_vac_nocomorbidities -> latent_early_ds_nocomorbidities [label=rate_force_weak_ds]
			susceptible_treated_nocomorbidities -> latent_early_ds_nocomorbidities [label=rate_force_weak_ds]
			latent_late_ds_nocomorbidities -> latent_early_ds_nocomorbidities [label=rate_force_weak_ds]
			latent_early_ds_nocomorbidities -> latent_late_ds_nocomorbidities [label=2.3]
			latent_early_ds_nocomorbidities -> active_smearpos_ds_nocomorbidities [label=0.20]
			latent_late_ds_nocomorbidities -> active_smearpos_ds_nocomorbidities [label=0.000342]
			active_smearpos_ds_nocomorbidities -> latent_late_ds_nocomorbidities [label=0.10]
			latent_early_ds_nocomorbidities -> active_smearneg_ds_nocomorbidities [label=0.07]
			latent_late_ds_nocomorbidities -> active_smearneg_ds_nocomorbidities [label=0.000114]
			active_smearneg_ds_nocomorbidities -> latent_late_ds_nocomorbidities [label=0.30]
			latent_early_ds_nocomorbidities -> active_extrapul_ds_nocomorbidities [label=0.07]
			latent_late_ds_nocomorbidities -> active_extrapul_ds_nocomorbidities [label=0.000114]
			active_extrapul_ds_nocomorbidities -> latent_late_ds_nocomorbidities [label=0.30]
			active_smearpos_ds_nocomorbidities -> detect_smearpos_ds_nocomorbidities [label=2.9]
			active_smearpos_ds_nocomorbidities -> missed_smearpos_ds_nocomorbidities [label=0.47]
			detect_smearpos_ds_nocomorbidities -> treatment_infect_smearpos_ds_nocomorbidities [label=25.6]
			missed_smearpos_ds_nocomorbidities -> active_smearpos_ds_nocomorbidities [label=0.27]
			active_smearneg_ds_nocomorbidities -> detect_smearneg_ds_nocomorbidities [label=2.9]
			active_smearneg_ds_nocomorbidities -> missed_smearneg_ds_nocomorbidities [label=0.47]
			detect_smearneg_ds_nocomorbidities -> treatment_infect_smearneg_ds_nocomorbidities [label=25.6]
			missed_smearneg_ds_nocomorbidities -> active_smearneg_ds_nocomorbidities [label=0.27]
			active_extrapul_ds_nocomorbidities -> detect_extrapul_ds_nocomorbidities [label=2.9]
			active_extrapul_ds_nocomorbidities -> missed_extrapul_ds_nocomorbidities [label=0.47]
			detect_extrapul_ds_nocomorbidities -> treatment_infect_extrapul_ds_nocomorbidities [label=25.6]
			missed_extrapul_ds_nocomorbidities -> active_extrapul_ds_nocomorbidities [label=0.27]
			treatment_infect_smearpos_ds_nocomorbidities -> treatment_noninfect_smearpos_ds_nocomorbidities [label=28.0]
			treatment_infect_smearpos_ds_nocomorbidities -> active_smearpos_ds_nocomorbidities [label=0.18]
			treatment_noninfect_smearpos_ds_nocomorbidities -> active_smearpos_ds_nocomorbidities [label=0.17]
			treatment_noninfect_smearpos_ds_nocomorbidities -> susceptible_treated_nocomorbidities [label=1.7]
			treatment_infect_smearneg_ds_nocomorbidities -> treatment_noninfect_smearneg_ds_nocomorbidities [label=28.0]
			treatment_infect_smearneg_ds_nocomorbidities -> active_smearneg_ds_nocomorbidities [label=0.18]
			treatment_noninfect_smearneg_ds_nocomorbidities -> active_smearneg_ds_nocomorbidities [label=0.17]
			treatment_noninfect_smearneg_ds_nocomorbidities -> susceptible_treated_nocomorbidities [label=1.7]
			treatment_infect_extrapul_ds_nocomorbidities -> treatment_noninfect_extrapul_ds_nocomorbidities [label=28.0]
			treatment_infect_extrapul_ds_nocomorbidities -> active_extrapul_ds_nocomorbidities [label=0.18]
			treatment_noninfect_extrapul_ds_nocomorbidities -> active_extrapul_ds_nocomorbidities [label=0.17]
			treatment_noninfect_extrapul_ds_nocomorbidities -> susceptible_treated_nocomorbidities [label=1.7]
			active_smearpos_ds_nocomorbidities -> tb_death [label=0.26]
			detect_smearpos_ds_nocomorbidities -> tb_death [label=0.26]
			active_smearneg_ds_nocomorbidities -> tb_death [label=0.07]
			detect_smearneg_ds_nocomorbidities -> tb_death [label=0.07]
			active_extrapul_ds_nocomorbidities -> tb_death [label=0.07]
			detect_extrapul_ds_nocomorbidities -> tb_death [label=0.07]
			treatment_infect_smearpos_ds_nocomorbidities -> tb_death [label=0.18]
			treatment_noninfect_smearpos_ds_nocomorbidities -> tb_death [label=0.17]
			treatment_infect_smearneg_ds_nocomorbidities -> tb_death [label=0.18]
			treatment_noninfect_smearneg_ds_nocomorbidities -> tb_death [label=0.17]
			treatment_infect_extrapul_ds_nocomorbidities -> tb_death [label=0.18]
			treatment_noninfect_extrapul_ds_nocomorbidities -> tb_death [label=0.17]
}