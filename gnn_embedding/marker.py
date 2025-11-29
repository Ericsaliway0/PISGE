from reactome2py import analysis


class Marker_ori:
    def __init__(self, marker_list, p_value):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        """Enrichment analysis performed on all the pathways.
        
        First all the hit pathways are obtained. Then, it is determined
        which of them are significant (p_value < threshold).

        Returns
        -------
        dict
            Dictionary of significant pathways, where stids are keys
            and the values stored are p_value and significance of
            each pathway
        """
        result = analysis.identifiers(ids=self.markers, interactors=False, page_size='1', page='1',
                                      species='Homo Sapiens', sort_by='ENTITIES_FDR', order='ASC',
                                      resource='TOTAL', p_value='1', include_disease=False, min_entities=None,
                                      max_entities=None, projection=True)
        token = result['summary']['token']
        token_result = analysis.token(token, species='Homo sapiens', page_size='-1', page='-1', sort_by='ENTITIES_FDR',
                                      order='ASC', resource='TOTAL', p_value='1', include_disease=False,
                                      min_entities=None, max_entities=None)
        info = [(p['stId'], p['entities']['pValue']) for p in token_result['pathways']]
        pathway_significance = {}
        for stid, p_val in info:
            significance = 'significant' if p_val < self.p_value else 'non-significant'
            pathway_significance[stid] = {'p_value': round(p_val, 4), 'significance': significance}
        return pathway_significance

import pandas as pd
from reactome2py import analysis


class Marker:
    def __init__(self, marker_list, p_value=0.05):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        # Submit marker IDs to Reactome Analysis
        result = analysis.identifiers(
            ids=self.markers,
            interactors=False,
            page_size='1',
            page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )

        token = result['summary']['token']

        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1',
            page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None
        )

        pathway_stats = {}
        for p in token_result['pathways']:
            stid = p['stId']
            name = p.get('displayName', '')

            p_val = p['entities']['pValue']
            fdr = p['entities'].get('fdr')
            found = p['entities'].get('found', 0)
            total = p['entities'].get('total', 0)

            # --- Extract hit genes safely ---
            entities_block = p['entities'].get('entities', [])
            interactors_block = p['entities'].get('interactors', [])
            hits = []

            for ent in entities_block:
                hits.append(ent.get('name', ent.get('identifier', '')))
            for inter in interactors_block:
                hits.append(inter.get('name', inter.get('identifier', '')))

            hit_genes_str = ",".join(sorted(set(hits))) if hits else ""

            significance = 'significant' if p_val < self.p_value else 'non-significant'
            hit_ratio = round(found / total, 4) if total else 0.0

            pathway_stats[stid] = {
                "stId": stid,
                "name": name,
                "p_value": round(p_val, 6),
                "fdr": round(fdr, 6) if fdr is not None else None,
                "found": found,
                "total": total,
                "hit_ratio": hit_ratio,
                "hit_genes": hit_genes_str,
                "significance": significance
            }

        return pathway_stats

class Marker:
    def __init__(self, marker_list, p_value):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

    def enrichment_analysis(self):
        """
        Enrichment analysis performed on all the pathways.

        Returns
        -------
        dict
            Dictionary of pathways, where stIds are keys and values are full
            metadata including p_value, significance, and all Reactome fields.
        """
        result = analysis.identifiers(
            ids=self.markers,
            interactors=False,
            page_size='1', page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',  # keep everything
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )

        token = result['summary']['token']
        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1', page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None
        )

        pathway_results = {}
        for p in token_result['pathways']:
            stid = p['stId']
            p_val = float(p['entities']['pValue'])

            # Add significance relative to your threshold
            significance = 'significant' if p_val < self.p_value else 'non-significant'

            # Copy everything from Reactome's response + extra fields
            pathway_results[stid] = {
                **p,  # keep ALL Reactome-provided info (name, species, entities, fdr, url, etc.)
                "p_value": round(p_val, 6),
                "significance": significance
            }

        return pathway_results

def save_pathway_stats_(pathway_results, save_dir="results/pathway_stats"):
    import os, pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    all_records = []
    for stid, data in pathway_results.items():
        # Flatten Reactome dict to only keep useful fields
        name = data.get("name", "")
        entities = data.get("entities", {})
        p_val = float(data.get("p_value", 1.0))
        fdr = entities.get("fdr", None)
        found = entities.get("found", 0)
        total = entities.get("total", 1)
        hit_ratio = found / total if total else 0
        hit_genes = entities.get("exp", [])
        hit_genes_str = ",".join(hit_genes) if hit_genes else ""
        significance = data.get("significance", "non-significant")

        record = {
            "stId": stid,
            "name": name,
            "p_value": round(p_val, 6),
            "fdr": round(fdr, 6) if fdr is not None else None,
            "found": found,
            "total": total,
            "hit_ratio": round(hit_ratio, 6),
            "hit_genes": ",".join(hit_genes) if hit_genes else "",
            "significance": significance
        }

        all_records.append(record)

        # Save single pathway CSV
        pd.DataFrame([record]).to_csv(
            os.path.join(save_dir, f"{stid}_stats.csv"), index=False
        )

    # Save combined CSV
    pd.DataFrame(all_records).to_csv(
        os.path.join(save_dir, "all_pathway_stats.csv"), index=False
    )
    print(f"✅ Saved {len(all_records)} pathway results to {save_dir}")

class Marker:
    def __init__(self, marker_list, p_value):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        self.result = self.enrichment_analysis()

        # Save enrichment results immediately
        save_pathway_stats(self.result, save_dir="embedding/results/pathway_stats")

    def enrichment_analysis(self):
        result = analysis.identifiers(
            ids=self.markers,
            interactors=False,
            page_size='1', page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )

        token = result['summary']['token']
        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1', page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None
        )

        pathway_results = {}
        for p in token_result['pathways']:
            stid = p['stId']
            p_val = float(p['entities']['pValue'])
            significance = 'significant' if p_val < self.p_value else 'non-significant'

            # Debug: print full entities block for the first few pathways
            print("\n=== Entities for pathway:", p['name'], f"({stid}) ===")
            print(p['entities'])

            pathway_results[stid] = {
                **p,
                "p_value": round(p_val, 6),
                "significance": significance,
                "hit_genes": p["entities"].get("exp", [])  # list of hit genes
            }

        return pathway_results

import pandas as pd

class Marker:
    def __init__(self, marker_list, p_value, pathways_mapping_file="data/processed/pathways_mapped_all_genes.tsv"):
        self.marker_list = marker_list
        self.markers = set(marker_list)
        self.p_value = p_value
        
        # Load pathway → genes mapping
        self.pathway_to_genes = self.load_pathway_mapping(pathways_mapping_file)
        
        # Run enrichment
        self.result = self.enrichment_analysis()

    def load_pathway_mapping(self, mapping_file):
        df = pd.read_csv(mapping_file, sep="\t", dtype=str)
        mapping = {}
        for _, row in df.iterrows():
            genes = [g for g in row[1:] if pd.notna(g)]
            mapping[row["PathwayID"]] = genes
        return mapping

    def enrichment_analysis(self):
        """
        Enrichment analysis performed on all the pathways.
        Returns a dict where stIds are keys and values include:
        - all Reactome info
        - p_value
        - significance
        - hit_genes (computed from exp or local mapping)
        """
        result = analysis.identifiers(
            ids=",".join(self.marker_list),
            interactors=False,
            page_size='1', page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )

        token = result['summary']['token']
        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1',
            page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
        )

        pathway_results = {}
        for p in token_result['pathways']:
            stid = p['stId']
            p_val = float(p['entities']['pValue'])
            significance = 'significant' if p_val < self.p_value else 'non-significant'
            
            # Try Reactome exp first
            hit_genes = p['entities'].get('exp', [])
            
            # Fallback: intersect input markers with local mapping
            if not hit_genes:
                hit_genes = list(self.markers.intersection(self.pathway_to_genes.get(stid, [])))
            
            hit_genes_str = ",".join(hit_genes)
            
            pathway_results[stid] = {
                **p,  # keep all Reactome fields
                "p_value": round(p_val, 6),
                "significance": significance,
                "hit_genes": hit_genes_str
            }

        return pathway_results

import os
import pandas as pd

def save_enrichment_results_(marker_obj, output_dir="embedding/results/enrichment", filename="enrichment_all.csv"):
    """
    Save enrichment results to CSV with columns:
    Marker, Pathway, p_value, significance, hit_genes, plus any other Reactome info.
    
    Parameters
    ----------
    marker_obj : Marker
        Instance of Marker class with .result populated
    output_dir : str
        Directory to save CSV
    filename : str
        Name of CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for stid, info in marker_obj.result.items():
        # hit_genes is already a comma-separated string
        for marker in marker_obj.marker_list:
            # Only include markers that are actually in this pathway
            if marker in info.get("hit_genes", "").split(","):
                record = {
                    "Marker": marker,
                    "Pathway": stid,
                    "p_value": info.get("p_value", 1.0),
                    "significance": info.get("significance", "non-significant"),
                    "hit_genes": info.get("hit_genes", "")
                }
                records.append(record)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"✅ Enrichment results saved to {csv_path}")
    return df

import os
import pandas as pd

import os
import pandas as pd
from reactome2py import analysis  # assuming you are using reactome2py


class Marker:
    def __init__(self, marker_list, p_value,
                 pathways_mapping_file="data/processed/pathways_mapped_all_genes.tsv",
                 save_dir="embedding/results/enrichment"):
        self.marker_list = marker_list
        self.markers = set(marker_list)
        self.p_value = p_value
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Load pathway → genes mapping
        self.pathway_to_genes = self.load_pathway_mapping(pathways_mapping_file)

        # Run enrichment
        self.result, self.pathway_stats = self.enrichment_analysis()

        # Save CSVs
        self.save_results_csv()

    def load_pathway_mapping(self, mapping_file):
        df = pd.read_csv(mapping_file, sep="\t", low_memory=False)
        mapping = {}
        for _, row in df.iterrows():
            genes = [g for g in row[1:] if pd.notna(g)]
            mapping[row["PathwayID"]] = genes
        return mapping

    def enrichment_analysis(self):
        # Run Reactome analysis
        result = analysis.identifiers(
            ids=",".join(self.marker_list),
            interactors=False,
            page_size='1', page='1',
            species='Homo Sapiens',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
            projection=True
        )
        token = result['summary']['token']
        token_result = analysis.token(
            token,
            species='Homo sapiens',
            page_size='-1',
            page='-1',
            sort_by='ENTITIES_FDR',
            order='ASC',
            resource='TOTAL',
            p_value='1',
            include_disease=False,
            min_entities=None,
            max_entities=None,
        )

        pathway_results = {}
        pathway_stats = {}

        for p in token_result['pathways']:
            stid = p['stId']
            name = p.get('name', "")
            entities = p['entities']
            p_val = float(entities.get('pValue', 1.0))
            fdr = entities.get('fdr')
            found = entities.get('found')
            total = entities.get('total')
            hit_ratio = entities.get('ratio')
            significance = 'significant' if p_val < self.p_value else 'non-significant'

            # Try Reactome exp first
            hit_genes = entities.get('exp', [])
            # Fallback: intersect input markers with local mapping
            if not hit_genes:
                hit_genes = list(self.markers.intersection(self.pathway_to_genes.get(stid, [])))
            hit_genes_str = ",".join(hit_genes)

            # Save full info
            pathway_results[stid] = {
                **p,
                "p_value": p_val,  # no rounding
                "significance": significance,
                "hit_genes": hit_genes_str
            }

            # Save stats for separate CSV
            pathway_stats[stid] = {
                "stId": stid,
                "name": name,
                "p_value": p_val,   # no rounding
                "fdr": fdr if fdr is not None else None,
                "found": found,
                "total": total,
                "hit_ratio": hit_ratio,
                "hit_genes": hit_genes_str,
                "significance": significance
            }

        return pathway_results, pathway_stats

    def save_results_csv(self):
        # Simple CSV (stId + p_value + significance)
        simple_csv = os.path.join(self.save_dir, "enrichment_simple.csv")
        simple_df = pd.DataFrame([
            {"stId": stid,
             "p_value": f"{info['p_value']:.6E}",
             "significance": info["significance"]}
            for stid, info in self.result.items()
        ])
        simple_df.to_csv(simple_csv, index=False)

        # Expanded CSV (all stats, with formatted p_value and fdr)
        expanded_csv = os.path.join(self.save_dir, "enrichment_expanded.csv")
        expanded_df = pd.DataFrame(self.pathway_stats.values())

        if "p_value" in expanded_df.columns:
            expanded_df["p_value"] = expanded_df["p_value"].apply(lambda x: f"{x:.6E}")
        if "fdr" in expanded_df.columns:
            expanded_df["fdr"] = expanded_df["fdr"].apply(lambda x: f"{x:.6E}" if pd.notna(x) else None)

        expanded_df.to_csv(expanded_csv, index=False)

        print(f"✅ Saved simple CSV: {simple_csv}")
        print(f"✅ Saved expanded CSV: {expanded_csv}")
