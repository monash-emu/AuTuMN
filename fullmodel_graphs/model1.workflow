digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbs
		susceptible_vac_nocomorbs
		susceptible_treated_nocomorbs
		latent_early_ds_nocomorbs
		latent_early_mdr_nocomorbs
		latent_late_ds_nocomorbs
		latent_late_mdr_nocomorbs
		active_smearpos_ds_nocomorbs
		active_smearpos_mdr_nocomorbs
		detect_smearpos_ds_nocomorbs
		detect_smearpos_mdr_nocomorbs
		missed_smearpos_ds_nocomorbs
		missed_smearpos_mdr_nocomorbs
		treatment_infect_smearpos_ds_nocomorbs
		treatment_infect_smearpos_mdr_nocomorbs
		treatment_noninfect_smearpos_ds_nocomorbs
		treatment_noninfect_smearpos_mdr_nocomorbs
		tb_death
			susceptible_fully_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_ds]
			susceptible_vac_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_treated_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			latent_late_ds_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_fully_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_mdr]
			susceptible_vac_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_treated_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			latent_late_mdr_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			active_smearpos_ds_nocomorbs -> missed_smearpos_ds_nocomorbs [label=program_rate_missed]
			active_smearpos_mdr_nocomorbs -> missed_smearpos_mdr_nocomorbs [label=program_rate_missed]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_nocomorbs [label=program_rate_detect]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_nocomorbs [label=program_rate_detect]
			latent_early_ds_nocomorbs -> latent_late_ds_nocomorbs [label=2.2]
			latent_early_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.30]
			latent_late_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.0070]
			active_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			missed_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			detect_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			latent_early_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=2.2]
			latent_early_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.30]
			latent_late_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.0070]
			active_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			missed_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			detect_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			missed_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=4.0]
			missed_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=4.0]
			treatment_infect_smearpos_ds_nocomorbs -> treatment_noninfect_smearpos_ds_nocomorbs [label=28.4]
			treatment_noninfect_smearpos_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.16]
			treatment_noninfect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.15]
			treatment_infect_smearpos_mdr_nocomorbs -> treatment_noninfect_smearpos_mdr_nocomorbs [label=11.8]
			treatment_noninfect_smearpos_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.15]
			active_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			missed_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			detect_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			active_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			missed_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			detect_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			treatment_infect_smearpos_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearpos_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_nocomorbs -> tb_death [label=0.0499]
}