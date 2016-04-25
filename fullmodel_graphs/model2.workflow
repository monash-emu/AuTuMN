digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbs
		susceptible_fully_hiv
		susceptible_fully_diabetes
		susceptible_vac_nocomorbs
		susceptible_vac_hiv
		susceptible_vac_diabetes
		susceptible_treated_nocomorbs
		susceptible_treated_hiv
		susceptible_treated_diabetes
		latent_early_ds_nocomorbs
		latent_early_mdr_nocomorbs
		latent_early_xdr_nocomorbs
		latent_early_ds_hiv
		latent_early_mdr_hiv
		latent_early_xdr_hiv
		latent_early_ds_diabetes
		latent_early_mdr_diabetes
		latent_early_xdr_diabetes
		latent_late_ds_nocomorbs
		latent_late_mdr_nocomorbs
		latent_late_xdr_nocomorbs
		latent_late_ds_hiv
		latent_late_mdr_hiv
		latent_late_xdr_hiv
		latent_late_ds_diabetes
		latent_late_mdr_diabetes
		latent_late_xdr_diabetes
		active_smearpos_ds_nocomorbs
		active_smearneg_ds_nocomorbs
		active_smearpos_mdr_nocomorbs
		active_smearneg_mdr_nocomorbs
		active_smearpos_xdr_nocomorbs
		active_smearneg_xdr_nocomorbs
		active_smearpos_ds_hiv
		active_smearneg_ds_hiv
		active_smearpos_mdr_hiv
		active_smearneg_mdr_hiv
		active_smearpos_xdr_hiv
		active_smearneg_xdr_hiv
		active_smearpos_ds_diabetes
		active_smearneg_ds_diabetes
		active_smearpos_mdr_diabetes
		active_smearneg_mdr_diabetes
		active_smearpos_xdr_diabetes
		active_smearneg_xdr_diabetes
		detect_smearpos_ds_nocomorbs
		detect_smearneg_ds_nocomorbs
		detect_smearpos_mdr_nocomorbs
		detect_smearneg_mdr_nocomorbs
		detect_smearpos_xdr_nocomorbs
		detect_smearneg_xdr_nocomorbs
		detect_smearpos_ds_hiv
		detect_smearneg_ds_hiv
		detect_smearpos_mdr_hiv
		detect_smearneg_mdr_hiv
		detect_smearpos_xdr_hiv
		detect_smearneg_xdr_hiv
		detect_smearpos_ds_diabetes
		detect_smearneg_ds_diabetes
		detect_smearpos_mdr_diabetes
		detect_smearneg_mdr_diabetes
		detect_smearpos_xdr_diabetes
		detect_smearneg_xdr_diabetes
		missed_smearpos_ds_nocomorbs
		missed_smearneg_ds_nocomorbs
		missed_smearpos_mdr_nocomorbs
		missed_smearneg_mdr_nocomorbs
		missed_smearpos_xdr_nocomorbs
		missed_smearneg_xdr_nocomorbs
		missed_smearpos_ds_hiv
		missed_smearneg_ds_hiv
		missed_smearpos_mdr_hiv
		missed_smearneg_mdr_hiv
		missed_smearpos_xdr_hiv
		missed_smearneg_xdr_hiv
		missed_smearpos_ds_diabetes
		missed_smearneg_ds_diabetes
		missed_smearpos_mdr_diabetes
		missed_smearneg_mdr_diabetes
		missed_smearpos_xdr_diabetes
		missed_smearneg_xdr_diabetes
		treatment_infect_smearpos_ds_nocomorbs
		treatment_infect_smearneg_ds_nocomorbs
		treatment_infect_smearpos_mdr_nocomorbs
		treatment_infect_smearneg_mdr_nocomorbs
		treatment_infect_smearpos_xdr_nocomorbs
		treatment_infect_smearneg_xdr_nocomorbs
		treatment_infect_smearpos_ds_hiv
		treatment_infect_smearneg_ds_hiv
		treatment_infect_smearpos_mdr_hiv
		treatment_infect_smearneg_mdr_hiv
		treatment_infect_smearpos_xdr_hiv
		treatment_infect_smearneg_xdr_hiv
		treatment_infect_smearpos_ds_diabetes
		treatment_infect_smearneg_ds_diabetes
		treatment_infect_smearpos_mdr_diabetes
		treatment_infect_smearneg_mdr_diabetes
		treatment_infect_smearpos_xdr_diabetes
		treatment_infect_smearneg_xdr_diabetes
		treatment_noninfect_smearpos_ds_nocomorbs
		treatment_noninfect_smearneg_ds_nocomorbs
		treatment_noninfect_smearpos_mdr_nocomorbs
		treatment_noninfect_smearneg_mdr_nocomorbs
		treatment_noninfect_smearpos_xdr_nocomorbs
		treatment_noninfect_smearneg_xdr_nocomorbs
		treatment_noninfect_smearpos_ds_hiv
		treatment_noninfect_smearneg_ds_hiv
		treatment_noninfect_smearpos_mdr_hiv
		treatment_noninfect_smearneg_mdr_hiv
		treatment_noninfect_smearpos_xdr_hiv
		treatment_noninfect_smearneg_xdr_hiv
		treatment_noninfect_smearpos_ds_diabetes
		treatment_noninfect_smearneg_ds_diabetes
		treatment_noninfect_smearpos_mdr_diabetes
		treatment_noninfect_smearneg_mdr_diabetes
		treatment_noninfect_smearpos_xdr_diabetes
		treatment_noninfect_smearneg_xdr_diabetes
		tb_death
			susceptible_fully_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_ds]
			susceptible_vac_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_treated_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			latent_late_ds_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_fully_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_mdr]
			susceptible_vac_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_treated_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			latent_late_mdr_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_fully_nocomorbs -> latent_early_xdr_nocomorbs [label=rate_force_xdr]
			susceptible_vac_nocomorbs -> latent_early_xdr_nocomorbs [label=rate_force_weak_xdr]
			susceptible_treated_nocomorbs -> latent_early_xdr_nocomorbs [label=rate_force_weak_xdr]
			latent_late_xdr_nocomorbs -> latent_early_xdr_nocomorbs [label=rate_force_weak_xdr]
			susceptible_fully_hiv -> latent_early_ds_hiv [label=rate_force_ds]
			susceptible_vac_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			susceptible_treated_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			latent_late_ds_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			susceptible_fully_hiv -> latent_early_mdr_hiv [label=rate_force_mdr]
			susceptible_vac_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			susceptible_treated_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			latent_late_mdr_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			susceptible_fully_hiv -> latent_early_xdr_hiv [label=rate_force_xdr]
			susceptible_vac_hiv -> latent_early_xdr_hiv [label=rate_force_weak_xdr]
			susceptible_treated_hiv -> latent_early_xdr_hiv [label=rate_force_weak_xdr]
			latent_late_xdr_hiv -> latent_early_xdr_hiv [label=rate_force_weak_xdr]
			susceptible_fully_diabetes -> latent_early_ds_diabetes [label=rate_force_ds]
			susceptible_vac_diabetes -> latent_early_ds_diabetes [label=rate_force_weak_ds]
			susceptible_treated_diabetes -> latent_early_ds_diabetes [label=rate_force_weak_ds]
			latent_late_ds_diabetes -> latent_early_ds_diabetes [label=rate_force_weak_ds]
			susceptible_fully_diabetes -> latent_early_mdr_diabetes [label=rate_force_mdr]
			susceptible_vac_diabetes -> latent_early_mdr_diabetes [label=rate_force_weak_mdr]
			susceptible_treated_diabetes -> latent_early_mdr_diabetes [label=rate_force_weak_mdr]
			latent_late_mdr_diabetes -> latent_early_mdr_diabetes [label=rate_force_weak_mdr]
			susceptible_fully_diabetes -> latent_early_xdr_diabetes [label=rate_force_xdr]
			susceptible_vac_diabetes -> latent_early_xdr_diabetes [label=rate_force_weak_xdr]
			susceptible_treated_diabetes -> latent_early_xdr_diabetes [label=rate_force_weak_xdr]
			latent_late_xdr_diabetes -> latent_early_xdr_diabetes [label=rate_force_weak_xdr]
			active_smearpos_ds_nocomorbs -> missed_smearpos_ds_nocomorbs [label=program_rate_missed]
			active_smearpos_ds_hiv -> missed_smearpos_ds_hiv [label=program_rate_missed]
			active_smearpos_ds_diabetes -> missed_smearpos_ds_diabetes [label=program_rate_missed]
			active_smearneg_ds_nocomorbs -> missed_smearneg_ds_nocomorbs [label=program_rate_missed]
			active_smearneg_ds_hiv -> missed_smearneg_ds_hiv [label=program_rate_missed]
			active_smearneg_ds_diabetes -> missed_smearneg_ds_diabetes [label=program_rate_missed]
			active_smearpos_mdr_nocomorbs -> missed_smearpos_mdr_nocomorbs [label=program_rate_missed]
			active_smearpos_mdr_hiv -> missed_smearpos_mdr_hiv [label=program_rate_missed]
			active_smearpos_mdr_diabetes -> missed_smearpos_mdr_diabetes [label=program_rate_missed]
			active_smearneg_mdr_nocomorbs -> missed_smearneg_mdr_nocomorbs [label=program_rate_missed]
			active_smearneg_mdr_hiv -> missed_smearneg_mdr_hiv [label=program_rate_missed]
			active_smearneg_mdr_diabetes -> missed_smearneg_mdr_diabetes [label=program_rate_missed]
			active_smearpos_xdr_nocomorbs -> missed_smearpos_xdr_nocomorbs [label=program_rate_missed]
			active_smearpos_xdr_hiv -> missed_smearpos_xdr_hiv [label=program_rate_missed]
			active_smearpos_xdr_diabetes -> missed_smearpos_xdr_diabetes [label=program_rate_missed]
			active_smearneg_xdr_nocomorbs -> missed_smearneg_xdr_nocomorbs [label=program_rate_missed]
			active_smearneg_xdr_hiv -> missed_smearneg_xdr_hiv [label=program_rate_missed]
			active_smearneg_xdr_diabetes -> missed_smearneg_xdr_diabetes [label=program_rate_missed]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_nocomorbs [label=program_rate_detect]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_nocomorbs [label=program_rate_detect]
			active_smearpos_xdr_nocomorbs -> detect_smearpos_xdr_nocomorbs [label=program_rate_detect]
			active_smearpos_ds_hiv -> detect_smearpos_ds_hiv [label=program_rate_detect]
			active_smearpos_mdr_hiv -> detect_smearpos_mdr_hiv [label=program_rate_detect]
			active_smearpos_xdr_hiv -> detect_smearpos_xdr_hiv [label=program_rate_detect]
			active_smearpos_ds_diabetes -> detect_smearpos_ds_diabetes [label=program_rate_detect]
			active_smearpos_mdr_diabetes -> detect_smearpos_mdr_diabetes [label=program_rate_detect]
			active_smearpos_xdr_diabetes -> detect_smearpos_xdr_diabetes [label=program_rate_detect]
			active_smearneg_ds_nocomorbs -> detect_smearneg_ds_nocomorbs [label=program_rate_detect]
			active_smearneg_mdr_nocomorbs -> detect_smearneg_mdr_nocomorbs [label=program_rate_detect]
			active_smearneg_xdr_nocomorbs -> detect_smearneg_xdr_nocomorbs [label=program_rate_detect]
			active_smearneg_ds_hiv -> detect_smearneg_ds_hiv [label=program_rate_detect]
			active_smearneg_mdr_hiv -> detect_smearneg_mdr_hiv [label=program_rate_detect]
			active_smearneg_xdr_hiv -> detect_smearneg_xdr_hiv [label=program_rate_detect]
			active_smearneg_ds_diabetes -> detect_smearneg_ds_diabetes [label=program_rate_detect]
			active_smearneg_mdr_diabetes -> detect_smearneg_mdr_diabetes [label=program_rate_detect]
			active_smearneg_xdr_diabetes -> detect_smearneg_xdr_diabetes [label=program_rate_detect]
			latent_early_ds_nocomorbs -> latent_late_ds_nocomorbs [label=2.2]
			latent_early_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.12]
			latent_late_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.0029]
			active_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			missed_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			detect_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.10]
			latent_early_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.18]
			latent_late_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.0041]
			active_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			missed_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			detect_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			latent_early_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=2.2]
			latent_early_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.12]
			latent_late_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.0029]
			active_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			missed_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			detect_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.10]
			latent_early_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.18]
			latent_late_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.0041]
			active_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			missed_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			detect_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			latent_early_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=2.2]
			latent_early_xdr_nocomorbs -> active_smearpos_xdr_nocomorbs [label=0.12]
			latent_late_xdr_nocomorbs -> active_smearpos_xdr_nocomorbs [label=0.0029]
			active_smearpos_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.10]
			missed_smearpos_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.10]
			detect_smearpos_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.10]
			latent_early_xdr_nocomorbs -> active_smearneg_xdr_nocomorbs [label=0.18]
			latent_late_xdr_nocomorbs -> active_smearneg_xdr_nocomorbs [label=0.0041]
			active_smearneg_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.27]
			missed_smearneg_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.27]
			detect_smearneg_xdr_nocomorbs -> latent_late_xdr_nocomorbs [label=0.27]
			latent_early_ds_hiv -> latent_late_ds_hiv [label=2.2]
			latent_early_ds_hiv -> active_smearpos_ds_hiv [label=0.12]
			latent_late_ds_hiv -> active_smearpos_ds_hiv [label=0.0029]
			active_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.10]
			missed_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.10]
			detect_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.10]
			latent_early_ds_hiv -> active_smearneg_ds_hiv [label=0.18]
			latent_late_ds_hiv -> active_smearneg_ds_hiv [label=0.0041]
			active_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			missed_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			detect_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			latent_early_mdr_hiv -> latent_late_mdr_hiv [label=2.2]
			latent_early_mdr_hiv -> active_smearpos_mdr_hiv [label=0.12]
			latent_late_mdr_hiv -> active_smearpos_mdr_hiv [label=0.0029]
			active_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.10]
			missed_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.10]
			detect_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.10]
			latent_early_mdr_hiv -> active_smearneg_mdr_hiv [label=0.18]
			latent_late_mdr_hiv -> active_smearneg_mdr_hiv [label=0.0041]
			active_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			missed_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			detect_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			latent_early_xdr_hiv -> latent_late_xdr_hiv [label=2.2]
			latent_early_xdr_hiv -> active_smearpos_xdr_hiv [label=0.12]
			latent_late_xdr_hiv -> active_smearpos_xdr_hiv [label=0.0029]
			active_smearpos_xdr_hiv -> latent_late_xdr_hiv [label=0.10]
			missed_smearpos_xdr_hiv -> latent_late_xdr_hiv [label=0.10]
			detect_smearpos_xdr_hiv -> latent_late_xdr_hiv [label=0.10]
			latent_early_xdr_hiv -> active_smearneg_xdr_hiv [label=0.18]
			latent_late_xdr_hiv -> active_smearneg_xdr_hiv [label=0.0041]
			active_smearneg_xdr_hiv -> latent_late_xdr_hiv [label=0.27]
			missed_smearneg_xdr_hiv -> latent_late_xdr_hiv [label=0.27]
			detect_smearneg_xdr_hiv -> latent_late_xdr_hiv [label=0.27]
			latent_early_ds_diabetes -> latent_late_ds_diabetes [label=2.2]
			latent_early_ds_diabetes -> active_smearpos_ds_diabetes [label=0.12]
			latent_late_ds_diabetes -> active_smearpos_ds_diabetes [label=0.0029]
			active_smearpos_ds_diabetes -> latent_late_ds_diabetes [label=0.10]
			missed_smearpos_ds_diabetes -> latent_late_ds_diabetes [label=0.10]
			detect_smearpos_ds_diabetes -> latent_late_ds_diabetes [label=0.10]
			latent_early_ds_diabetes -> active_smearneg_ds_diabetes [label=0.18]
			latent_late_ds_diabetes -> active_smearneg_ds_diabetes [label=0.0041]
			active_smearneg_ds_diabetes -> latent_late_ds_diabetes [label=0.27]
			missed_smearneg_ds_diabetes -> latent_late_ds_diabetes [label=0.27]
			detect_smearneg_ds_diabetes -> latent_late_ds_diabetes [label=0.27]
			latent_early_mdr_diabetes -> latent_late_mdr_diabetes [label=2.2]
			latent_early_mdr_diabetes -> active_smearpos_mdr_diabetes [label=0.12]
			latent_late_mdr_diabetes -> active_smearpos_mdr_diabetes [label=0.0029]
			active_smearpos_mdr_diabetes -> latent_late_mdr_diabetes [label=0.10]
			missed_smearpos_mdr_diabetes -> latent_late_mdr_diabetes [label=0.10]
			detect_smearpos_mdr_diabetes -> latent_late_mdr_diabetes [label=0.10]
			latent_early_mdr_diabetes -> active_smearneg_mdr_diabetes [label=0.18]
			latent_late_mdr_diabetes -> active_smearneg_mdr_diabetes [label=0.0041]
			active_smearneg_mdr_diabetes -> latent_late_mdr_diabetes [label=0.27]
			missed_smearneg_mdr_diabetes -> latent_late_mdr_diabetes [label=0.27]
			detect_smearneg_mdr_diabetes -> latent_late_mdr_diabetes [label=0.27]
			latent_early_xdr_diabetes -> latent_late_xdr_diabetes [label=2.2]
			latent_early_xdr_diabetes -> active_smearpos_xdr_diabetes [label=0.12]
			latent_late_xdr_diabetes -> active_smearpos_xdr_diabetes [label=0.0029]
			active_smearpos_xdr_diabetes -> latent_late_xdr_diabetes [label=0.10]
			missed_smearpos_xdr_diabetes -> latent_late_xdr_diabetes [label=0.10]
			detect_smearpos_xdr_diabetes -> latent_late_xdr_diabetes [label=0.10]
			latent_early_xdr_diabetes -> active_smearneg_xdr_diabetes [label=0.18]
			latent_late_xdr_diabetes -> active_smearneg_xdr_diabetes [label=0.0041]
			active_smearneg_xdr_diabetes -> latent_late_xdr_diabetes [label=0.27]
			missed_smearneg_xdr_diabetes -> latent_late_xdr_diabetes [label=0.27]
			detect_smearneg_xdr_diabetes -> latent_late_xdr_diabetes [label=0.27]
			missed_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=4.0]
			missed_smearpos_ds_hiv -> active_smearpos_ds_hiv [label=4.0]
			missed_smearpos_ds_diabetes -> active_smearpos_ds_diabetes [label=4.0]
			missed_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=4.0]
			missed_smearneg_ds_hiv -> active_smearneg_ds_hiv [label=4.0]
			missed_smearneg_ds_diabetes -> active_smearneg_ds_diabetes [label=4.0]
			missed_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=4.0]
			missed_smearpos_mdr_hiv -> active_smearpos_mdr_hiv [label=4.0]
			missed_smearpos_mdr_diabetes -> active_smearpos_mdr_diabetes [label=4.0]
			missed_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=4.0]
			missed_smearneg_mdr_hiv -> active_smearneg_mdr_hiv [label=4.0]
			missed_smearneg_mdr_diabetes -> active_smearneg_mdr_diabetes [label=4.0]
			missed_smearpos_xdr_nocomorbs -> active_smearpos_xdr_nocomorbs [label=4.0]
			missed_smearpos_xdr_hiv -> active_smearpos_xdr_hiv [label=4.0]
			missed_smearpos_xdr_diabetes -> active_smearpos_xdr_diabetes [label=4.0]
			missed_smearneg_xdr_nocomorbs -> active_smearneg_xdr_nocomorbs [label=4.0]
			missed_smearneg_xdr_hiv -> active_smearneg_xdr_hiv [label=4.0]
			missed_smearneg_xdr_diabetes -> active_smearneg_xdr_diabetes [label=4.0]
			treatment_infect_smearpos_ds_nocomorbs -> treatment_noninfect_smearpos_ds_nocomorbs [label=28.4]
			treatment_noninfect_smearpos_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.16]
			treatment_noninfect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.15]
			treatment_infect_smearneg_ds_nocomorbs -> treatment_noninfect_smearneg_ds_nocomorbs [label=28.4]
			treatment_noninfect_smearneg_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.16]
			treatment_noninfect_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.15]
			treatment_infect_smearpos_mdr_nocomorbs -> treatment_noninfect_smearpos_mdr_nocomorbs [label=11.8]
			treatment_noninfect_smearpos_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.15]
			treatment_infect_smearneg_mdr_nocomorbs -> treatment_noninfect_smearneg_mdr_nocomorbs [label=11.8]
			treatment_noninfect_smearneg_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.15]
			treatment_infect_smearpos_xdr_nocomorbs -> treatment_noninfect_smearpos_xdr_nocomorbs [label=5.8]
			treatment_noninfect_smearpos_xdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.16]
			treatment_infect_smearpos_xdr_nocomorbs -> active_smearpos_xdr_nocomorbs [label=0.20]
			treatment_noninfect_smearpos_xdr_nocomorbs -> active_smearpos_xdr_nocomorbs [label=0.15]
			treatment_infect_smearneg_xdr_nocomorbs -> treatment_noninfect_smearneg_xdr_nocomorbs [label=5.8]
			treatment_noninfect_smearneg_xdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.16]
			treatment_infect_smearneg_xdr_nocomorbs -> active_smearneg_xdr_nocomorbs [label=0.20]
			treatment_noninfect_smearneg_xdr_nocomorbs -> active_smearneg_xdr_nocomorbs [label=0.15]
			treatment_infect_smearpos_ds_hiv -> treatment_noninfect_smearpos_ds_hiv [label=28.4]
			treatment_noninfect_smearpos_ds_hiv -> susceptible_treated_hiv [label=2.0]
			treatment_infect_smearpos_ds_hiv -> active_smearpos_ds_hiv [label=0.16]
			treatment_noninfect_smearpos_ds_hiv -> active_smearpos_ds_hiv [label=0.15]
			treatment_infect_smearneg_ds_hiv -> treatment_noninfect_smearneg_ds_hiv [label=28.4]
			treatment_noninfect_smearneg_ds_hiv -> susceptible_treated_hiv [label=2.0]
			treatment_infect_smearneg_ds_hiv -> active_smearneg_ds_hiv [label=0.16]
			treatment_noninfect_smearneg_ds_hiv -> active_smearneg_ds_hiv [label=0.15]
			treatment_infect_smearpos_mdr_hiv -> treatment_noninfect_smearpos_mdr_hiv [label=11.8]
			treatment_noninfect_smearpos_mdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearpos_mdr_hiv -> active_smearpos_mdr_hiv [label=0.18]
			treatment_noninfect_smearpos_mdr_hiv -> active_smearpos_mdr_hiv [label=0.15]
			treatment_infect_smearneg_mdr_hiv -> treatment_noninfect_smearneg_mdr_hiv [label=11.8]
			treatment_noninfect_smearneg_mdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearneg_mdr_hiv -> active_smearneg_mdr_hiv [label=0.18]
			treatment_noninfect_smearneg_mdr_hiv -> active_smearneg_mdr_hiv [label=0.15]
			treatment_infect_smearpos_xdr_hiv -> treatment_noninfect_smearpos_xdr_hiv [label=5.8]
			treatment_noninfect_smearpos_xdr_hiv -> susceptible_treated_hiv [label=0.16]
			treatment_infect_smearpos_xdr_hiv -> active_smearpos_xdr_hiv [label=0.20]
			treatment_noninfect_smearpos_xdr_hiv -> active_smearpos_xdr_hiv [label=0.15]
			treatment_infect_smearneg_xdr_hiv -> treatment_noninfect_smearneg_xdr_hiv [label=5.8]
			treatment_noninfect_smearneg_xdr_hiv -> susceptible_treated_hiv [label=0.16]
			treatment_infect_smearneg_xdr_hiv -> active_smearneg_xdr_hiv [label=0.20]
			treatment_noninfect_smearneg_xdr_hiv -> active_smearneg_xdr_hiv [label=0.15]
			treatment_infect_smearpos_ds_diabetes -> treatment_noninfect_smearpos_ds_diabetes [label=28.4]
			treatment_noninfect_smearpos_ds_diabetes -> susceptible_treated_diabetes [label=2.0]
			treatment_infect_smearpos_ds_diabetes -> active_smearpos_ds_diabetes [label=0.16]
			treatment_noninfect_smearpos_ds_diabetes -> active_smearpos_ds_diabetes [label=0.15]
			treatment_infect_smearneg_ds_diabetes -> treatment_noninfect_smearneg_ds_diabetes [label=28.4]
			treatment_noninfect_smearneg_ds_diabetes -> susceptible_treated_diabetes [label=2.0]
			treatment_infect_smearneg_ds_diabetes -> active_smearneg_ds_diabetes [label=0.16]
			treatment_noninfect_smearneg_ds_diabetes -> active_smearneg_ds_diabetes [label=0.15]
			treatment_infect_smearpos_mdr_diabetes -> treatment_noninfect_smearpos_mdr_diabetes [label=11.8]
			treatment_noninfect_smearpos_mdr_diabetes -> susceptible_treated_diabetes [label=0.32]
			treatment_infect_smearpos_mdr_diabetes -> active_smearpos_mdr_diabetes [label=0.18]
			treatment_noninfect_smearpos_mdr_diabetes -> active_smearpos_mdr_diabetes [label=0.15]
			treatment_infect_smearneg_mdr_diabetes -> treatment_noninfect_smearneg_mdr_diabetes [label=11.8]
			treatment_noninfect_smearneg_mdr_diabetes -> susceptible_treated_diabetes [label=0.32]
			treatment_infect_smearneg_mdr_diabetes -> active_smearneg_mdr_diabetes [label=0.18]
			treatment_noninfect_smearneg_mdr_diabetes -> active_smearneg_mdr_diabetes [label=0.15]
			treatment_infect_smearpos_xdr_diabetes -> treatment_noninfect_smearpos_xdr_diabetes [label=5.8]
			treatment_noninfect_smearpos_xdr_diabetes -> susceptible_treated_diabetes [label=0.16]
			treatment_infect_smearpos_xdr_diabetes -> active_smearpos_xdr_diabetes [label=0.20]
			treatment_noninfect_smearpos_xdr_diabetes -> active_smearpos_xdr_diabetes [label=0.15]
			treatment_infect_smearneg_xdr_diabetes -> treatment_noninfect_smearneg_xdr_diabetes [label=5.8]
			treatment_noninfect_smearneg_xdr_diabetes -> susceptible_treated_diabetes [label=0.16]
			treatment_infect_smearneg_xdr_diabetes -> active_smearneg_xdr_diabetes [label=0.20]
			treatment_noninfect_smearneg_xdr_diabetes -> active_smearneg_xdr_diabetes [label=0.15]
			active_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			missed_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			detect_smearpos_ds_nocomorbs -> tb_death [label=0.23]
			active_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			missed_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			active_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			missed_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			detect_smearpos_mdr_nocomorbs -> tb_death [label=0.23]
			active_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			missed_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			active_smearpos_xdr_nocomorbs -> tb_death [label=0.23]
			missed_smearpos_xdr_nocomorbs -> tb_death [label=0.23]
			detect_smearpos_xdr_nocomorbs -> tb_death [label=0.23]
			active_smearneg_xdr_nocomorbs -> tb_death [label=0.07]
			missed_smearneg_xdr_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_xdr_nocomorbs -> tb_death [label=0.07]
			active_smearpos_ds_hiv -> tb_death [label=0.23]
			missed_smearpos_ds_hiv -> tb_death [label=0.23]
			detect_smearpos_ds_hiv -> tb_death [label=0.23]
			active_smearneg_ds_hiv -> tb_death [label=0.07]
			missed_smearneg_ds_hiv -> tb_death [label=0.07]
			detect_smearneg_ds_hiv -> tb_death [label=0.07]
			active_smearpos_mdr_hiv -> tb_death [label=0.23]
			missed_smearpos_mdr_hiv -> tb_death [label=0.23]
			detect_smearpos_mdr_hiv -> tb_death [label=0.23]
			active_smearneg_mdr_hiv -> tb_death [label=0.07]
			missed_smearneg_mdr_hiv -> tb_death [label=0.07]
			detect_smearneg_mdr_hiv -> tb_death [label=0.07]
			active_smearpos_xdr_hiv -> tb_death [label=0.23]
			missed_smearpos_xdr_hiv -> tb_death [label=0.23]
			detect_smearpos_xdr_hiv -> tb_death [label=0.23]
			active_smearneg_xdr_hiv -> tb_death [label=0.07]
			missed_smearneg_xdr_hiv -> tb_death [label=0.07]
			detect_smearneg_xdr_hiv -> tb_death [label=0.07]
			active_smearpos_ds_diabetes -> tb_death [label=0.23]
			missed_smearpos_ds_diabetes -> tb_death [label=0.23]
			detect_smearpos_ds_diabetes -> tb_death [label=0.23]
			active_smearneg_ds_diabetes -> tb_death [label=0.07]
			missed_smearneg_ds_diabetes -> tb_death [label=0.07]
			detect_smearneg_ds_diabetes -> tb_death [label=0.07]
			active_smearpos_mdr_diabetes -> tb_death [label=0.23]
			missed_smearpos_mdr_diabetes -> tb_death [label=0.23]
			detect_smearpos_mdr_diabetes -> tb_death [label=0.23]
			active_smearneg_mdr_diabetes -> tb_death [label=0.07]
			missed_smearneg_mdr_diabetes -> tb_death [label=0.07]
			detect_smearneg_mdr_diabetes -> tb_death [label=0.07]
			active_smearpos_xdr_diabetes -> tb_death [label=0.23]
			missed_smearpos_xdr_diabetes -> tb_death [label=0.23]
			detect_smearpos_xdr_diabetes -> tb_death [label=0.23]
			active_smearneg_xdr_diabetes -> tb_death [label=0.07]
			missed_smearneg_xdr_diabetes -> tb_death [label=0.07]
			detect_smearneg_xdr_diabetes -> tb_death [label=0.07]
			treatment_infect_smearpos_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearneg_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearpos_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearneg_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearpos_xdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_xdr_nocomorbs -> tb_death [label=0.0498]
			treatment_infect_smearneg_xdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_xdr_nocomorbs -> tb_death [label=0.0498]
			treatment_infect_smearpos_ds_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_hiv -> tb_death [label=0.0500]
			treatment_infect_smearneg_ds_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_hiv -> tb_death [label=0.0500]
			treatment_infect_smearpos_mdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_hiv -> tb_death [label=0.0499]
			treatment_infect_smearneg_mdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_hiv -> tb_death [label=0.0499]
			treatment_infect_smearpos_xdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_xdr_hiv -> tb_death [label=0.0498]
			treatment_infect_smearneg_xdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_xdr_hiv -> tb_death [label=0.0498]
			treatment_infect_smearpos_ds_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_diabetes -> tb_death [label=0.0500]
			treatment_infect_smearneg_ds_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_diabetes -> tb_death [label=0.0500]
			treatment_infect_smearpos_mdr_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_diabetes -> tb_death [label=0.0499]
			treatment_infect_smearneg_mdr_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_diabetes -> tb_death [label=0.0499]
			treatment_infect_smearpos_xdr_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearpos_xdr_diabetes -> tb_death [label=0.0498]
			treatment_infect_smearneg_xdr_diabetes -> tb_death [label=0.05]
			treatment_noninfect_smearneg_xdr_diabetes -> tb_death [label=0.0498]
}