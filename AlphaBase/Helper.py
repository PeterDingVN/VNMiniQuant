
class StandardizedDataDict(dict):

    def __init__(self, raw_data: dict, symbol_configs: list):
        super().__init__()
        self._raw_data = raw_data
        self._configs = symbol_configs
        self._alias_map = {}
        self._build_standardized_maps()

    def _normalize_symbol(self, symbol_str: str) -> str:
        """Applies your custom normalization rules to extract the base symbol string."""
        base = symbol_str.upper().strip()
        
        # Strip prefixes if they match your known array
        suf = base.split(":", 1)[0]
        if suf in ['VN', 'CP', 'C&M', 'VNF']:
            base = base.split(":", 1)[1]
            
        # Clean filesystem-unsafe characters
        return base.replace("/", "_").replace(":", "_")

    def _build_standardized_maps(self):

        for cfg in self._configs:
            orig_sym = cfg.get("original_symbol", "")
            target_tf = cfg.get("target_interval", "")
            
            norm_sym = self._normalize_symbol(orig_sym)
            standardized_key = f"{norm_sym}_{target_tf}"

            matched_raw_key = None
            for raw_key in self._raw_data.keys():
                if orig_sym.lower() in raw_key.lower():
                    matched_raw_key = raw_key
                    break
            
            if matched_raw_key:
                df = self._raw_data[matched_raw_key]
                
                self[standardized_key] = df
                
                self._alias_map[orig_sym.upper()] = standardized_key
                self._alias_map[orig_sym.lower()] = standardized_key
                self._alias_map[norm_sym] = standardized_key

    def __getitem__(self, key: str):
        if not isinstance(key, str):
            return super().__getitem__(key)
            
        clean_key = key.strip()
        
        # Rule 1: Direct hit if they pass the exact exact internal rule (e.g., 'BSI_1D')
        if clean_key in self:
            return super().__getitem__(clean_key)
            
        # Rule 2: Hit via configuration string aliases (e.g., 'vn:bSi', 'VN:BSI', 'BSI')
        if clean_key in self._alias_map:
            return super().__getitem__(self._alias_map[clean_key])
        if clean_key.upper() in self._alias_map:
            return super().__getitem__(self._alias_map[clean_key.upper()])
        if clean_key.lower() in self._alias_map:
            return super().__getitem__(self._alias_map[clean_key.lower()])
            
        # Rule 3: Dynamic normalization fallback
        norm_key = self._normalize_symbol(clean_key)
        for internal_key in self.keys():
            if internal_key.startswith(norm_key):
                return super().__getitem__(internal_key)
                
        raise KeyError(f"Symbol variation '{key}' could not be resolved to any loaded dataset.")

