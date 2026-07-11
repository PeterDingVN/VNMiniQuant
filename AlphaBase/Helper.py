import re


class StandardizedDataDict(dict):

    def __init__(self, raw_data: dict, symbol_configs: list):
        super().__init__()
        self._raw_data = raw_data
        self._configs = symbol_configs
        self._alias_map = {}
        self._build_standardized_maps()

    def _normalize_symbol(self, symbol_str: str) -> str:
        """Normalize symbols so aliases such as 'VN:BSI' and 'vn_bsi_5m' resolve consistently."""
        if not isinstance(symbol_str, str):
            return str(symbol_str)

        base = symbol_str.strip().upper().replace("/", "_").replace(":", "_")
        base = re.sub(r"[_\s]+", "_", base)
        parts = [part for part in base.split("_") if part]

        if parts and parts[0] in {"VN", "CP", "C&M", "VNF"}:
            parts = parts[1:]

        return "_".join(parts).lower()

    def _build_standardized_maps(self):
        for cfg in self._configs:
            orig_sym = cfg.get("original_symbol", "")
            target_tf = cfg.get("target_interval", "")

            norm_sym = self._normalize_symbol(orig_sym)
            norm_tf = self._normalize_symbol(target_tf)
            standardized_key = f"{norm_sym}_{norm_tf}"

            matched_raw_key = None
            for raw_key in self._raw_data.keys():
                if orig_sym.lower() in raw_key.lower():
                    matched_raw_key = raw_key
                    break

            if matched_raw_key:
                df = self._raw_data[matched_raw_key]
                self[standardized_key] = df

                for alias in {
                    orig_sym,
                    orig_sym.upper(),
                    orig_sym.lower(),
                    norm_sym,
                    f"{orig_sym}_{target_tf}",
                    f"{norm_sym}_{norm_tf}",
                    matched_raw_key,
                    standardized_key,
                }:
                    if alias:
                        self._alias_map[self._normalize_symbol(alias)] = standardized_key

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            return super().__getitem__(key)

        clean_key = key.strip()

        if dict.__contains__(self, clean_key):
            return super().__getitem__(clean_key)

        norm_key = self._normalize_symbol(clean_key)
        if norm_key in self._alias_map:
            return super().__getitem__(self._alias_map[norm_key])

        for internal_key in self.keys():
            if self._normalize_symbol(internal_key) == norm_key or self._normalize_symbol(internal_key).startswith(norm_key):
                return super().__getitem__(internal_key)

        raise KeyError(f"Unable to resolve '{key}', please check if {key} is in config file 'data'")

    def __contains__(self, key):
        if not isinstance(key, str):
            return dict.__contains__(self, key)

        if dict.__contains__(self, key):
            return True

        norm_key = self._normalize_symbol(key)
        if norm_key in self._alias_map:
            return True

        for internal_key in self.keys():
            if self._normalize_symbol(internal_key) == norm_key or self._normalize_symbol(internal_key).startswith(norm_key):
                return True

        return False