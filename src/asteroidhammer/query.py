"""Query services"""
import numpy as np
from tqdm import tqdm
import tess_ephem
import pyia


def query_asteroid_dict(tpf, max_objs=None, magnitude_lower_limit=100, magnitude_upper_limit=0):
    mask = np.in1d(np.arange(len(tpf)), np.arange(0, len(tpf) * 4, 4))
    objs = tpf.query_solar_system_objects(cadence_mask=mask)
    objs = objs.drop_duplicates("Name")
    objs = objs[(objs.Mv < magnitude_lower_limit) & (objs.Mv > magnitude_upper_limit)]
    objs = objs.sort_values('Mv').reset_index(drop=True)
    if max_objs is not None:
        objs = objs.head(max_objs)

    names = np.asarray(objs.sort_values('Mv').Name)
    ephs = []
    for name in tqdm(names):
        eph = tess_ephem.ephem(name, tpf.astropy_time)
        eph = eph[(eph.sector == tpf.sector) & (eph.camera == tpf.camera) & (eph.ccd == tpf.ccd)]
        ephs.append(eph)

    c = np.zeros((len(tpf), (len(ephs)))) * np.nan
    r = np.zeros((len(tpf), (len(ephs)))) * np.nan
    for jdx, eph in enumerate(ephs):
        c[np.in1d(tpf.astropy_time.jd, np.asarray([i.jd for i in eph.index])), jdx] = np.asarray(eph.column)
        r[np.in1d(tpf.astropy_time.jd, np.asarray([i.jd for i in eph.index])), jdx] = np.asarray(eph.row)

    c -= tpf.column
    r -= tpf.row
    return {'time':tpf.astropy_time.jd, 'column':c, 'row':r, 'names':names, 'Mv':np.asarray(objs['Mv'])}


@functools.lru_cache()
def query_gaia_sources(tpf, magnitude_limit=18):
    epoch = int(Time(tpf.time.mean() + tpf.get_header()['BJDREFI'], format="jd").isot[:4])
    r = r = np.hypot(*np.diff(tpf.wcs.wcs_pix2world([[tpf.column, tpf.row], [tpf.column + tpf.shape[1]/2, tpf.row + tpf.shape[1]/2]], 0), axis=0)[0])
    ras, decs, rads = [tpf.ra], [tpf.dec], [r]
    def _get_gaia(ras, decs, rads, epoch, magnitude_limit):
        wheres = [
            f"""1=CONTAINS(
                      POINT('ICRS',ra,dec),
                      CIRCLE('ICRS',{ra},{dec},{rad}))"""
            for ra, dec, rad in zip(ras, decs, rads)
        ]

        where = """\n\tOR """.join(wheres)
        gd = pyia.GaiaData.from_query(
            f"""SELECT solution_id, designation, source_id, random_index, ref_epoch,
            coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
            parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error,
            ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
            dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr,
            parallax_pmdec_corr, pmra_pmdec_corr, astrometric_n_obs_al,
            astrometric_n_obs_ac, astrometric_n_good_obs_al, astrometric_n_bad_obs_al,
            astrometric_gof_al, astrometric_chi2_al, astrometric_excess_noise,
            astrometric_excess_noise_sig, astrometric_params_solved,
            astrometric_primary_flag, astrometric_weight_al, astrometric_pseudo_colour,
            astrometric_pseudo_colour_error, mean_varpi_factor_al,
            astrometric_matched_observations, visibility_periods_used,
            astrometric_sigma5d_max, frame_rotator_object_type, matched_observations,
            duplicated_source, phot_g_n_obs, phot_g_mean_flux, phot_g_mean_flux_error,
            phot_g_mean_flux_over_error, phot_g_mean_mag, phot_bp_n_obs,
            phot_bp_mean_flux, phot_bp_mean_flux_error, phot_bp_mean_flux_over_error,
            phot_bp_mean_mag, phot_rp_n_obs, phot_rp_mean_flux, phot_rp_mean_flux_error,
            phot_rp_mean_flux_over_error, phot_rp_mean_mag, phot_bp_rp_excess_factor,
            phot_proc_mode, bp_rp, bp_g, g_rp, radial_velocity, radial_velocity_error,
            rv_nb_transits, rv_template_teff, rv_template_logg, rv_template_fe_h,
            phot_variable_flag, l, b, ecl_lon, ecl_lat, priam_flags, teff_val,
            teff_percentile_lower, teff_percentile_upper, a_g_val,
            a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val,
            e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags,
            radius_val, radius_percentile_lower, radius_percentile_upper, lum_val,
            lum_percentile_lower, lum_percentile_upper, datalink_url,
            epoch_photometry_url, ra as ra_gaia, dec as dec_gaia FROM (
     SELECT *,
     EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
     FROM gaiadr2.gaia_source
     WHERE {where}
    )  AS subquery
    WHERE phot_g_mean_mag<={magnitude_limit}
    """
        )
        return gd
    return _get_gaia(ras, decs, rads, epoch, magnitude_limit)
