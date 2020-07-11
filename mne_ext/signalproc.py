import mne

def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.

    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)