digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbs
		susceptible_fully_hiv
		susceptible_vac_nocomorbs
		susceptible_vac_hiv
		susceptible_treated_nocomorbs
		susceptible_treated_hiv
		latent_early_ds_nocomorbs
		latent_early_mdr_nocomorbs
		latent_early_ds_hiv
		latent_early_mdr_hiv
		latent_late_ds_nocomorbs
		latent_late_mdr_nocomorbs
		latent_late_ds_hiv
		latent_late_mdr_hiv
		active_smearpos_ds_nocomorbs
		active_smearneg_ds_nocomorbs
		active_extrapul_ds_nocomorbs
		active_smearpos_mdr_nocomorbs
		active_smearneg_mdr_nocomorbs
		active_extrapul_mdr_nocomorbs
		active_smearpos_ds_hiv
		active_smearneg_ds_hiv
		active_extrapul_ds_hiv
		active_smearpos_mdr_hiv
		active_smearneg_mdr_hiv
		active_extrapul_mdr_hiv
		detect_smearpos_ds_asds_nocomorbs
		detect_smearpos_ds_asmdr_nocomorbs
		detect_smearneg_ds_asds_nocomorbs
		detect_smearneg_ds_asmdr_nocomorbs
		detect_extrapul_ds_asds_nocomorbs
		detect_extrapul_ds_asmdr_nocomorbs
		detect_smearpos_mdr_asds_nocomorbs
		detect_smearpos_mdr_asmdr_nocomorbs
		detect_smearneg_mdr_asds_nocomorbs
		detect_smearneg_mdr_asmdr_nocomorbs
		detect_extrapul_mdr_asds_nocomorbs
		detect_extrapul_mdr_asmdr_nocomorbs
		detect_smearpos_ds_asds_hiv
		detect_smearpos_ds_asmdr_hiv
		detect_smearneg_ds_asds_hiv
		detect_smearneg_ds_asmdr_hiv
		detect_extrapul_ds_asds_hiv
		detect_extrapul_ds_asmdr_hiv
		detect_smearpos_mdr_asds_hiv
		detect_smearpos_mdr_asmdr_hiv
		detect_smearneg_mdr_asds_hiv
		detect_smearneg_mdr_asmdr_hiv
		detect_extrapul_mdr_asds_hiv
		detect_extrapul_mdr_asmdr_hiv
		missed_smearpos_ds_nocomorbs
		missed_smearneg_ds_nocomorbs
		missed_extrapul_ds_nocomorbs
		missed_smearpos_mdr_nocomorbs
		missed_smearneg_mdr_nocomorbs
		missed_extrapul_mdr_nocomorbs
		missed_smearpos_ds_hiv
		missed_smearneg_ds_hiv
		missed_extrapul_ds_hiv
		missed_smearpos_mdr_hiv
		missed_smearneg_mdr_hiv
		missed_extrapul_mdr_hiv
		treatment_infect_smearpos_ds_asds_nocomorbs
		treatment_infect_smearpos_ds_asmdr_nocomorbs
		treatment_infect_smearneg_ds_asds_nocomorbs
		treatment_infect_smearneg_ds_asmdr_nocomorbs
		treatment_infect_extrapul_ds_asds_nocomorbs
		treatment_infect_extrapul_ds_asmdr_nocomorbs
		treatment_infect_smearpos_mdr_asds_nocomorbs
		treatment_infect_smearpos_mdr_asmdr_nocomorbs
		treatment_infect_smearneg_mdr_asds_nocomorbs
		treatment_infect_smearneg_mdr_asmdr_nocomorbs
		treatment_infect_extrapul_mdr_asds_nocomorbs
		treatment_infect_extrapul_mdr_asmdr_nocomorbs
		treatment_infect_smearpos_ds_asds_hiv
		treatment_infect_smearpos_ds_asmdr_hiv
		treatment_infect_smearneg_ds_asds_hiv
		treatment_infect_smearneg_ds_asmdr_hiv
		treatment_infect_extrapul_ds_asds_hiv
		treatment_infect_extrapul_ds_asmdr_hiv
		treatment_infect_smearpos_mdr_asds_hiv
		treatment_infect_smearpos_mdr_asmdr_hiv
		treatment_infect_smearneg_mdr_asds_hiv
		treatment_infect_smearneg_mdr_asmdr_hiv
		treatment_infect_extrapul_mdr_asds_hiv
		treatment_infect_extrapul_mdr_asmdr_hiv
		treatment_noninfect_smearpos_ds_asds_nocomorbs
		treatment_noninfect_smearpos_ds_asmdr_nocomorbs
		treatment_noninfect_smearneg_ds_asds_nocomorbs
		treatment_noninfect_smearneg_ds_asmdr_nocomorbs
		treatment_noninfect_extrapul_ds_asds_nocomorbs
		treatment_noninfect_extrapul_ds_asmdr_nocomorbs
		treatment_noninfect_smearpos_mdr_asds_nocomorbs
		treatment_noninfect_smearpos_mdr_asmdr_nocomorbs
		treatment_noninfect_smearneg_mdr_asds_nocomorbs
		treatment_noninfect_smearneg_mdr_asmdr_nocomorbs
		treatment_noninfect_extrapul_mdr_asds_nocomorbs
		treatment_noninfect_extrapul_mdr_asmdr_nocomorbs
		treatment_noninfect_smearpos_ds_asds_hiv
		treatment_noninfect_smearpos_ds_asmdr_hiv
		treatment_noninfect_smearneg_ds_asds_hiv
		treatment_noninfect_smearneg_ds_asmdr_hiv
		treatment_noninfect_extrapul_ds_asds_hiv
		treatment_noninfect_extrapul_ds_asmdr_hiv
		treatment_noninfect_smearpos_mdr_asds_hiv
		treatment_noninfect_smearpos_mdr_asmdr_hiv
		treatment_noninfect_smearneg_mdr_asds_hiv
		treatment_noninfect_smearneg_mdr_asmdr_hiv
		treatment_noninfect_extrapul_mdr_asds_hiv
		treatment_noninfect_extrapul_mdr_asmdr_hiv
		lowquality_smearpos_ds_nocomorbs
		lowquality_smearneg_ds_nocomorbs
		lowquality_extrapul_ds_nocomorbs
		lowquality_smearpos_mdr_nocomorbs
		lowquality_smearneg_mdr_nocomorbs
		lowquality_extrapul_mdr_nocomorbs
		lowquality_smearpos_ds_hiv
		lowquality_smearneg_ds_hiv
		lowquality_extrapul_ds_hiv
		lowquality_smearpos_mdr_hiv
		lowquality_smearneg_mdr_hiv
		lowquality_extrapul_mdr_hiv
		tb_death
			susceptible_fully_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_ds]
			susceptible_vac_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_treated_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			latent_late_ds_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_fully_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_mdr]
			susceptible_vac_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_treated_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			latent_late_mdr_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_fully_hiv -> latent_early_ds_hiv [label=rate_force_ds]
			susceptible_vac_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			susceptible_treated_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			latent_late_ds_hiv -> latent_early_ds_hiv [label=rate_force_weak_ds]
			susceptible_fully_hiv -> latent_early_mdr_hiv [label=rate_force_mdr]
			susceptible_vac_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			susceptible_treated_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			latent_late_mdr_hiv -> latent_early_mdr_hiv [label=rate_force_weak_mdr]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_asds_nocomorbs [label=program_rate_detect_ds_asds]
			active_smearneg_ds_nocomorbs -> detect_smearneg_ds_asds_nocomorbs [label=program_rate_detect_ds_asds]
			active_extrapul_ds_nocomorbs -> detect_extrapul_ds_asds_nocomorbs [label=program_rate_detect_ds_asds]
			active_smearpos_ds_hiv -> detect_smearpos_ds_asds_hiv [label=program_rate_detect_ds_asds]
			active_smearneg_ds_hiv -> detect_smearneg_ds_asds_hiv [label=program_rate_detect_ds_asds]
			active_extrapul_ds_hiv -> detect_extrapul_ds_asds_hiv [label=program_rate_detect_ds_asds]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_asds_nocomorbs [label=program_rate_detect_mdr_asds]
			active_smearneg_mdr_nocomorbs -> detect_smearneg_mdr_asds_nocomorbs [label=program_rate_detect_mdr_asds]
			active_extrapul_mdr_nocomorbs -> detect_extrapul_mdr_asds_nocomorbs [label=program_rate_detect_mdr_asds]
			active_smearpos_mdr_hiv -> detect_smearpos_mdr_asds_hiv [label=program_rate_detect_mdr_asds]
			active_smearneg_mdr_hiv -> detect_smearneg_mdr_asds_hiv [label=program_rate_detect_mdr_asds]
			active_extrapul_mdr_hiv -> detect_extrapul_mdr_asds_hiv [label=program_rate_detect_mdr_asds]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_asmdr_nocomorbs [label=program_rate_detect_mdr_asmdr]
			active_smearneg_mdr_nocomorbs -> detect_smearneg_mdr_asmdr_nocomorbs [label=program_rate_detect_mdr_asmdr]
			active_extrapul_mdr_nocomorbs -> detect_extrapul_mdr_asmdr_nocomorbs [label=program_rate_detect_mdr_asmdr]
			active_smearpos_mdr_hiv -> detect_smearpos_mdr_asmdr_hiv [label=program_rate_detect_mdr_asmdr]
			active_smearneg_mdr_hiv -> detect_smearneg_mdr_asmdr_hiv [label=program_rate_detect_mdr_asmdr]
			active_extrapul_mdr_hiv -> detect_extrapul_mdr_asmdr_hiv [label=program_rate_detect_mdr_asmdr]
			active_smearpos_ds_nocomorbs -> missed_smearpos_ds_nocomorbs [label=program_rate_missed]
			active_smearneg_ds_nocomorbs -> missed_smearneg_ds_nocomorbs [label=program_rate_missed]
			active_extrapul_ds_nocomorbs -> missed_extrapul_ds_nocomorbs [label=program_rate_missed]
			active_smearpos_mdr_nocomorbs -> missed_smearpos_mdr_nocomorbs [label=program_rate_missed]
			active_smearneg_mdr_nocomorbs -> missed_smearneg_mdr_nocomorbs [label=program_rate_missed]
			active_extrapul_mdr_nocomorbs -> missed_extrapul_mdr_nocomorbs [label=program_rate_missed]
			active_smearpos_ds_hiv -> missed_smearpos_ds_hiv [label=program_rate_missed]
			active_smearneg_ds_hiv -> missed_smearneg_ds_hiv [label=program_rate_missed]
			active_extrapul_ds_hiv -> missed_extrapul_ds_hiv [label=program_rate_missed]
			active_smearpos_mdr_hiv -> missed_smearpos_mdr_hiv [label=program_rate_missed]
			active_smearneg_mdr_hiv -> missed_smearneg_mdr_hiv [label=program_rate_missed]
			active_extrapul_mdr_hiv -> missed_extrapul_mdr_hiv [label=program_rate_missed]
			treatment_infect_smearpos_ds_asds_nocomorbs -> active_smearpos_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearpos_ds_asds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_smearpos_ds_asmdr_nocomorbs -> active_smearpos_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_smearpos_ds_asmdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_mdr]
			treatment_infect_smearneg_ds_asds_nocomorbs -> active_smearneg_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearneg_ds_asds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_smearneg_ds_asmdr_nocomorbs -> active_smearneg_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_smearneg_ds_asmdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_mdr]
			treatment_infect_extrapul_ds_asds_nocomorbs -> active_extrapul_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_extrapul_ds_asds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_extrapul_ds_asmdr_nocomorbs -> active_extrapul_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_extrapul_ds_asmdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_mdr]
			treatment_infect_smearpos_ds_asds_hiv -> active_smearpos_ds_hiv [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearpos_ds_asds_hiv -> active_smearpos_mdr_hiv [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_smearpos_ds_asmdr_hiv -> active_smearpos_ds_hiv [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_smearpos_ds_asmdr_hiv -> active_smearpos_mdr_hiv [label=program_rate_default_noninfect_amplify_mdr]
			treatment_infect_smearneg_ds_asds_hiv -> active_smearneg_ds_hiv [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearneg_ds_asds_hiv -> active_smearneg_mdr_hiv [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_smearneg_ds_asmdr_hiv -> active_smearneg_ds_hiv [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_smearneg_ds_asmdr_hiv -> active_smearneg_mdr_hiv [label=program_rate_default_noninfect_amplify_mdr]
			treatment_infect_extrapul_ds_asds_hiv -> active_extrapul_ds_hiv [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_extrapul_ds_asds_hiv -> active_extrapul_mdr_hiv [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_extrapul_ds_asmdr_hiv -> active_extrapul_ds_hiv [label=program_rate_default_noninfect_noamplify_mdr]
			treatment_infect_extrapul_ds_asmdr_hiv -> active_extrapul_mdr_hiv [label=program_rate_default_noninfect_amplify_mdr]
			latent_early_ds_nocomorbs -> latent_late_ds_nocomorbs [label=2.2]
			latent_early_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.12]
			latent_late_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.0029]
			active_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.13]
			missed_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.13]
			lowquality_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.13]
			detect_smearpos_ds_asds_nocomorbs -> latent_late_ds_nocomorbs [label=0.13]
			detect_smearpos_ds_asmdr_nocomorbs -> latent_late_ds_nocomorbs [label=0.13]
			latent_early_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.17]
			latent_late_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.0040]
			active_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			missed_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			lowquality_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			detect_smearneg_ds_asds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			detect_smearneg_ds_asmdr_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			latent_early_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.0051]
			latent_late_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.000120]
			active_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			missed_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			lowquality_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			detect_extrapul_ds_asds_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			detect_extrapul_ds_asmdr_nocomorbs -> latent_late_ds_nocomorbs [label=0.27]
			latent_early_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=2.2]
			latent_early_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.12]
			latent_late_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.0029]
			active_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.13]
			missed_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.13]
			lowquality_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.13]
			detect_smearpos_mdr_asds_nocomorbs -> latent_late_mdr_nocomorbs [label=0.13]
			detect_smearpos_mdr_asmdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.13]
			latent_early_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.17]
			latent_late_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.0040]
			active_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			missed_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			lowquality_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			detect_smearneg_mdr_asds_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			detect_smearneg_mdr_asmdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			latent_early_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.0051]
			latent_late_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.000120]
			active_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			missed_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			lowquality_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			detect_extrapul_mdr_asds_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			detect_extrapul_mdr_asmdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.27]
			latent_early_ds_hiv -> latent_late_ds_hiv [label=2.2]
			latent_early_ds_hiv -> active_smearpos_ds_hiv [label=0.12]
			latent_late_ds_hiv -> active_smearpos_ds_hiv [label=0.0029]
			active_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.13]
			missed_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.13]
			lowquality_smearpos_ds_hiv -> latent_late_ds_hiv [label=0.13]
			detect_smearpos_ds_asds_hiv -> latent_late_ds_hiv [label=0.13]
			detect_smearpos_ds_asmdr_hiv -> latent_late_ds_hiv [label=0.13]
			latent_early_ds_hiv -> active_smearneg_ds_hiv [label=0.17]
			latent_late_ds_hiv -> active_smearneg_ds_hiv [label=0.0040]
			active_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			missed_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			lowquality_smearneg_ds_hiv -> latent_late_ds_hiv [label=0.27]
			detect_smearneg_ds_asds_hiv -> latent_late_ds_hiv [label=0.27]
			detect_smearneg_ds_asmdr_hiv -> latent_late_ds_hiv [label=0.27]
			latent_early_ds_hiv -> active_extrapul_ds_hiv [label=0.0051]
			latent_late_ds_hiv -> active_extrapul_ds_hiv [label=0.000120]
			active_extrapul_ds_hiv -> latent_late_ds_hiv [label=0.27]
			missed_extrapul_ds_hiv -> latent_late_ds_hiv [label=0.27]
			lowquality_extrapul_ds_hiv -> latent_late_ds_hiv [label=0.27]
			detect_extrapul_ds_asds_hiv -> latent_late_ds_hiv [label=0.27]
			detect_extrapul_ds_asmdr_hiv -> latent_late_ds_hiv [label=0.27]
			latent_early_mdr_hiv -> latent_late_mdr_hiv [label=2.2]
			latent_early_mdr_hiv -> active_smearpos_mdr_hiv [label=0.12]
			latent_late_mdr_hiv -> active_smearpos_mdr_hiv [label=0.0029]
			active_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.13]
			missed_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.13]
			lowquality_smearpos_mdr_hiv -> latent_late_mdr_hiv [label=0.13]
			detect_smearpos_mdr_asds_hiv -> latent_late_mdr_hiv [label=0.13]
			detect_smearpos_mdr_asmdr_hiv -> latent_late_mdr_hiv [label=0.13]
			latent_early_mdr_hiv -> active_smearneg_mdr_hiv [label=0.17]
			latent_late_mdr_hiv -> active_smearneg_mdr_hiv [label=0.0040]
			active_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			missed_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			lowquality_smearneg_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			detect_smearneg_mdr_asds_hiv -> latent_late_mdr_hiv [label=0.27]
			detect_smearneg_mdr_asmdr_hiv -> latent_late_mdr_hiv [label=0.27]
			latent_early_mdr_hiv -> active_extrapul_mdr_hiv [label=0.0051]
			latent_late_mdr_hiv -> active_extrapul_mdr_hiv [label=0.000120]
			active_extrapul_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			missed_extrapul_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			lowquality_extrapul_mdr_hiv -> latent_late_mdr_hiv [label=0.27]
			detect_extrapul_mdr_asds_hiv -> latent_late_mdr_hiv [label=0.27]
			detect_extrapul_mdr_asmdr_hiv -> latent_late_mdr_hiv [label=0.27]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_asmdr_nocomorbs [label=0.0]
			active_smearneg_ds_nocomorbs -> detect_smearneg_ds_asmdr_nocomorbs [label=0.0]
			active_extrapul_ds_nocomorbs -> detect_extrapul_ds_asmdr_nocomorbs [label=0.0]
			active_smearpos_ds_hiv -> detect_smearpos_ds_asmdr_hiv [label=0.0]
			active_smearneg_ds_hiv -> detect_smearneg_ds_asmdr_hiv [label=0.0]
			active_extrapul_ds_hiv -> detect_extrapul_ds_asmdr_hiv [label=0.0]
			missed_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=4.0]
			detect_smearpos_ds_asds_nocomorbs -> treatment_infect_smearpos_ds_asds_nocomorbs [label=26.0]
			detect_smearpos_ds_asmdr_nocomorbs -> treatment_infect_smearpos_ds_asmdr_nocomorbs [label=26.0]
			missed_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=4.0]
			detect_smearneg_ds_asds_nocomorbs -> treatment_infect_smearneg_ds_asds_nocomorbs [label=26.0]
			detect_smearneg_ds_asmdr_nocomorbs -> treatment_infect_smearneg_ds_asmdr_nocomorbs [label=26.0]
			missed_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=4.0]
			detect_extrapul_ds_asds_nocomorbs -> treatment_infect_extrapul_ds_asds_nocomorbs [label=26.0]
			detect_extrapul_ds_asmdr_nocomorbs -> treatment_infect_extrapul_ds_asmdr_nocomorbs [label=26.0]
			missed_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=4.0]
			detect_smearpos_mdr_asds_nocomorbs -> treatment_infect_smearpos_mdr_asds_nocomorbs [label=26.0]
			detect_smearpos_mdr_asmdr_nocomorbs -> treatment_infect_smearpos_mdr_asmdr_nocomorbs [label=26.0]
			missed_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=4.0]
			detect_smearneg_mdr_asds_nocomorbs -> treatment_infect_smearneg_mdr_asds_nocomorbs [label=26.0]
			detect_smearneg_mdr_asmdr_nocomorbs -> treatment_infect_smearneg_mdr_asmdr_nocomorbs [label=26.0]
			missed_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=4.0]
			detect_extrapul_mdr_asds_nocomorbs -> treatment_infect_extrapul_mdr_asds_nocomorbs [label=26.0]
			detect_extrapul_mdr_asmdr_nocomorbs -> treatment_infect_extrapul_mdr_asmdr_nocomorbs [label=26.0]
			missed_smearpos_ds_hiv -> active_smearpos_ds_hiv [label=4.0]
			detect_smearpos_ds_asds_hiv -> treatment_infect_smearpos_ds_asds_hiv [label=26.0]
			detect_smearpos_ds_asmdr_hiv -> treatment_infect_smearpos_ds_asmdr_hiv [label=26.0]
			missed_smearneg_ds_hiv -> active_smearneg_ds_hiv [label=4.0]
			detect_smearneg_ds_asds_hiv -> treatment_infect_smearneg_ds_asds_hiv [label=26.0]
			detect_smearneg_ds_asmdr_hiv -> treatment_infect_smearneg_ds_asmdr_hiv [label=26.0]
			missed_extrapul_ds_hiv -> active_extrapul_ds_hiv [label=4.0]
			detect_extrapul_ds_asds_hiv -> treatment_infect_extrapul_ds_asds_hiv [label=26.0]
			detect_extrapul_ds_asmdr_hiv -> treatment_infect_extrapul_ds_asmdr_hiv [label=26.0]
			missed_smearpos_mdr_hiv -> active_smearpos_mdr_hiv [label=4.0]
			detect_smearpos_mdr_asds_hiv -> treatment_infect_smearpos_mdr_asds_hiv [label=26.0]
			detect_smearpos_mdr_asmdr_hiv -> treatment_infect_smearpos_mdr_asmdr_hiv [label=26.0]
			missed_smearneg_mdr_hiv -> active_smearneg_mdr_hiv [label=4.0]
			detect_smearneg_mdr_asds_hiv -> treatment_infect_smearneg_mdr_asds_hiv [label=26.0]
			detect_smearneg_mdr_asmdr_hiv -> treatment_infect_smearneg_mdr_asmdr_hiv [label=26.0]
			missed_extrapul_mdr_hiv -> active_extrapul_mdr_hiv [label=4.0]
			detect_extrapul_mdr_asds_hiv -> treatment_infect_extrapul_mdr_asds_hiv [label=26.0]
			detect_extrapul_mdr_asmdr_hiv -> treatment_infect_extrapul_mdr_asmdr_hiv [label=26.0]
			treatment_infect_smearpos_ds_asds_nocomorbs -> treatment_noninfect_smearpos_ds_asds_nocomorbs [label=28.4]
			treatment_noninfect_smearpos_ds_asds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearpos_ds_asmdr_nocomorbs -> treatment_noninfect_smearpos_ds_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_smearpos_ds_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearpos_mdr_asds_nocomorbs -> treatment_noninfect_smearpos_mdr_asds_nocomorbs [label=0.22]
			treatment_noninfect_smearpos_mdr_asds_nocomorbs -> susceptible_treated_nocomorbs [label=0.8]
			treatment_infect_smearpos_mdr_asds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.21]
			treatment_noninfect_smearpos_mdr_asds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.14]
			treatment_infect_smearpos_mdr_asmdr_nocomorbs -> treatment_noninfect_smearpos_mdr_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_smearpos_mdr_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearpos_mdr_asmdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearpos_mdr_asmdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.15]
			treatment_infect_smearneg_ds_asds_nocomorbs -> treatment_noninfect_smearneg_ds_asds_nocomorbs [label=28.4]
			treatment_noninfect_smearneg_ds_asds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearneg_ds_asmdr_nocomorbs -> treatment_noninfect_smearneg_ds_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_smearneg_ds_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearneg_mdr_asds_nocomorbs -> treatment_noninfect_smearneg_mdr_asds_nocomorbs [label=0.22]
			treatment_noninfect_smearneg_mdr_asds_nocomorbs -> susceptible_treated_nocomorbs [label=0.8]
			treatment_infect_smearneg_mdr_asds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.21]
			treatment_noninfect_smearneg_mdr_asds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.14]
			treatment_infect_smearneg_mdr_asmdr_nocomorbs -> treatment_noninfect_smearneg_mdr_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_smearneg_mdr_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearneg_mdr_asmdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearneg_mdr_asmdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.15]
			treatment_infect_extrapul_ds_asds_nocomorbs -> treatment_noninfect_extrapul_ds_asds_nocomorbs [label=28.4]
			treatment_noninfect_extrapul_ds_asds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_extrapul_ds_asmdr_nocomorbs -> treatment_noninfect_extrapul_ds_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_extrapul_ds_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_extrapul_mdr_asds_nocomorbs -> treatment_noninfect_extrapul_mdr_asds_nocomorbs [label=0.22]
			treatment_noninfect_extrapul_mdr_asds_nocomorbs -> susceptible_treated_nocomorbs [label=0.8]
			treatment_infect_extrapul_mdr_asds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.21]
			treatment_noninfect_extrapul_mdr_asds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.14]
			treatment_infect_extrapul_mdr_asmdr_nocomorbs -> treatment_noninfect_extrapul_mdr_asmdr_nocomorbs [label=11.8]
			treatment_noninfect_extrapul_mdr_asmdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_extrapul_mdr_asmdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.18]
			treatment_noninfect_extrapul_mdr_asmdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.15]
			treatment_infect_smearpos_ds_asds_hiv -> treatment_noninfect_smearpos_ds_asds_hiv [label=28.4]
			treatment_noninfect_smearpos_ds_asds_hiv -> susceptible_treated_hiv [label=2.0]
			treatment_infect_smearpos_ds_asmdr_hiv -> treatment_noninfect_smearpos_ds_asmdr_hiv [label=11.8]
			treatment_noninfect_smearpos_ds_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearpos_mdr_asds_hiv -> treatment_noninfect_smearpos_mdr_asds_hiv [label=0.22]
			treatment_noninfect_smearpos_mdr_asds_hiv -> susceptible_treated_hiv [label=0.8]
			treatment_infect_smearpos_mdr_asds_hiv -> active_smearpos_mdr_hiv [label=0.21]
			treatment_noninfect_smearpos_mdr_asds_hiv -> active_smearpos_mdr_hiv [label=0.14]
			treatment_infect_smearpos_mdr_asmdr_hiv -> treatment_noninfect_smearpos_mdr_asmdr_hiv [label=11.8]
			treatment_noninfect_smearpos_mdr_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearpos_mdr_asmdr_hiv -> active_smearpos_mdr_hiv [label=0.18]
			treatment_noninfect_smearpos_mdr_asmdr_hiv -> active_smearpos_mdr_hiv [label=0.15]
			treatment_infect_smearneg_ds_asds_hiv -> treatment_noninfect_smearneg_ds_asds_hiv [label=28.4]
			treatment_noninfect_smearneg_ds_asds_hiv -> susceptible_treated_hiv [label=2.0]
			treatment_infect_smearneg_ds_asmdr_hiv -> treatment_noninfect_smearneg_ds_asmdr_hiv [label=11.8]
			treatment_noninfect_smearneg_ds_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearneg_mdr_asds_hiv -> treatment_noninfect_smearneg_mdr_asds_hiv [label=0.22]
			treatment_noninfect_smearneg_mdr_asds_hiv -> susceptible_treated_hiv [label=0.8]
			treatment_infect_smearneg_mdr_asds_hiv -> active_smearneg_mdr_hiv [label=0.21]
			treatment_noninfect_smearneg_mdr_asds_hiv -> active_smearneg_mdr_hiv [label=0.14]
			treatment_infect_smearneg_mdr_asmdr_hiv -> treatment_noninfect_smearneg_mdr_asmdr_hiv [label=11.8]
			treatment_noninfect_smearneg_mdr_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_smearneg_mdr_asmdr_hiv -> active_smearneg_mdr_hiv [label=0.18]
			treatment_noninfect_smearneg_mdr_asmdr_hiv -> active_smearneg_mdr_hiv [label=0.15]
			treatment_infect_extrapul_ds_asds_hiv -> treatment_noninfect_extrapul_ds_asds_hiv [label=28.4]
			treatment_noninfect_extrapul_ds_asds_hiv -> susceptible_treated_hiv [label=2.0]
			treatment_infect_extrapul_ds_asmdr_hiv -> treatment_noninfect_extrapul_ds_asmdr_hiv [label=11.8]
			treatment_noninfect_extrapul_ds_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_extrapul_mdr_asds_hiv -> treatment_noninfect_extrapul_mdr_asds_hiv [label=0.22]
			treatment_noninfect_extrapul_mdr_asds_hiv -> susceptible_treated_hiv [label=0.8]
			treatment_infect_extrapul_mdr_asds_hiv -> active_extrapul_mdr_hiv [label=0.21]
			treatment_noninfect_extrapul_mdr_asds_hiv -> active_extrapul_mdr_hiv [label=0.14]
			treatment_infect_extrapul_mdr_asmdr_hiv -> treatment_noninfect_extrapul_mdr_asmdr_hiv [label=11.8]
			treatment_noninfect_extrapul_mdr_asmdr_hiv -> susceptible_treated_hiv [label=0.32]
			treatment_infect_extrapul_mdr_asmdr_hiv -> active_extrapul_mdr_hiv [label=0.18]
			treatment_noninfect_extrapul_mdr_asmdr_hiv -> active_extrapul_mdr_hiv [label=0.15]
			active_smearpos_ds_nocomorbs -> tb_death [label=0.20]
			missed_smearpos_ds_nocomorbs -> tb_death [label=0.20]
			lowquality_smearpos_ds_nocomorbs -> tb_death [label=0.20]
			detect_smearpos_ds_asds_nocomorbs -> tb_death [label=0.20]
			detect_smearpos_ds_asmdr_nocomorbs -> tb_death [label=0.20]
			active_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			missed_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			lowquality_smearneg_ds_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_ds_asds_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_ds_asmdr_nocomorbs -> tb_death [label=0.07]
			active_extrapul_ds_nocomorbs -> tb_death [label=0.07]
			missed_extrapul_ds_nocomorbs -> tb_death [label=0.07]
			lowquality_extrapul_ds_nocomorbs -> tb_death [label=0.07]
			detect_extrapul_ds_asds_nocomorbs -> tb_death [label=0.07]
			detect_extrapul_ds_asmdr_nocomorbs -> tb_death [label=0.07]
			active_smearpos_mdr_nocomorbs -> tb_death [label=0.20]
			missed_smearpos_mdr_nocomorbs -> tb_death [label=0.20]
			lowquality_smearpos_mdr_nocomorbs -> tb_death [label=0.20]
			detect_smearpos_mdr_asds_nocomorbs -> tb_death [label=0.20]
			detect_smearpos_mdr_asmdr_nocomorbs -> tb_death [label=0.20]
			active_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			missed_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			lowquality_smearneg_mdr_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_mdr_asds_nocomorbs -> tb_death [label=0.07]
			detect_smearneg_mdr_asmdr_nocomorbs -> tb_death [label=0.07]
			active_extrapul_mdr_nocomorbs -> tb_death [label=0.07]
			missed_extrapul_mdr_nocomorbs -> tb_death [label=0.07]
			lowquality_extrapul_mdr_nocomorbs -> tb_death [label=0.07]
			detect_extrapul_mdr_asds_nocomorbs -> tb_death [label=0.07]
			detect_extrapul_mdr_asmdr_nocomorbs -> tb_death [label=0.07]
			active_smearpos_ds_hiv -> tb_death [label=0.20]
			missed_smearpos_ds_hiv -> tb_death [label=0.20]
			lowquality_smearpos_ds_hiv -> tb_death [label=0.20]
			detect_smearpos_ds_asds_hiv -> tb_death [label=0.20]
			detect_smearpos_ds_asmdr_hiv -> tb_death [label=0.20]
			active_smearneg_ds_hiv -> tb_death [label=0.07]
			missed_smearneg_ds_hiv -> tb_death [label=0.07]
			lowquality_smearneg_ds_hiv -> tb_death [label=0.07]
			detect_smearneg_ds_asds_hiv -> tb_death [label=0.07]
			detect_smearneg_ds_asmdr_hiv -> tb_death [label=0.07]
			active_extrapul_ds_hiv -> tb_death [label=0.07]
			missed_extrapul_ds_hiv -> tb_death [label=0.07]
			lowquality_extrapul_ds_hiv -> tb_death [label=0.07]
			detect_extrapul_ds_asds_hiv -> tb_death [label=0.07]
			detect_extrapul_ds_asmdr_hiv -> tb_death [label=0.07]
			active_smearpos_mdr_hiv -> tb_death [label=0.20]
			missed_smearpos_mdr_hiv -> tb_death [label=0.20]
			lowquality_smearpos_mdr_hiv -> tb_death [label=0.20]
			detect_smearpos_mdr_asds_hiv -> tb_death [label=0.20]
			detect_smearpos_mdr_asmdr_hiv -> tb_death [label=0.20]
			active_smearneg_mdr_hiv -> tb_death [label=0.07]
			missed_smearneg_mdr_hiv -> tb_death [label=0.07]
			lowquality_smearneg_mdr_hiv -> tb_death [label=0.07]
			detect_smearneg_mdr_asds_hiv -> tb_death [label=0.07]
			detect_smearneg_mdr_asmdr_hiv -> tb_death [label=0.07]
			active_extrapul_mdr_hiv -> tb_death [label=0.07]
			missed_extrapul_mdr_hiv -> tb_death [label=0.07]
			lowquality_extrapul_mdr_hiv -> tb_death [label=0.07]
			detect_extrapul_mdr_asds_hiv -> tb_death [label=0.07]
			detect_extrapul_mdr_asmdr_hiv -> tb_death [label=0.07]
			treatment_infect_smearpos_ds_asds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_asds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearpos_ds_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearpos_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_noninfect_smearpos_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_infect_smearpos_mdr_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearneg_ds_asds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_asds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearneg_ds_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearneg_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_noninfect_smearneg_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_infect_smearneg_mdr_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_extrapul_ds_asds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_extrapul_ds_asds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_extrapul_ds_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_extrapul_ds_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_extrapul_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_noninfect_extrapul_mdr_asds_nocomorbs -> tb_death [label=0.06]
			treatment_infect_extrapul_mdr_asmdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_extrapul_mdr_asmdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearpos_ds_asds_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_asds_hiv -> tb_death [label=0.0500]
			treatment_infect_smearpos_ds_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_asmdr_hiv -> tb_death [label=0.0499]
			treatment_infect_smearpos_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_noninfect_smearpos_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_infect_smearpos_mdr_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_asmdr_hiv -> tb_death [label=0.0499]
			treatment_infect_smearneg_ds_asds_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_asds_hiv -> tb_death [label=0.0500]
			treatment_infect_smearneg_ds_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_asmdr_hiv -> tb_death [label=0.0499]
			treatment_infect_smearneg_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_noninfect_smearneg_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_infect_smearneg_mdr_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_asmdr_hiv -> tb_death [label=0.0499]
			treatment_infect_extrapul_ds_asds_hiv -> tb_death [label=0.05]
			treatment_noninfect_extrapul_ds_asds_hiv -> tb_death [label=0.0500]
			treatment_infect_extrapul_ds_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_extrapul_ds_asmdr_hiv -> tb_death [label=0.0499]
			treatment_infect_extrapul_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_noninfect_extrapul_mdr_asds_hiv -> tb_death [label=0.06]
			treatment_infect_extrapul_mdr_asmdr_hiv -> tb_death [label=0.05]
			treatment_noninfect_extrapul_mdr_asmdr_hiv -> tb_death [label=0.0499]
}