def _vprint(verbose: bool, *args, **kwargs) -> None:
    """Print only if verbose is True."""
    if verbose:
        print(*args, **kwargs)

