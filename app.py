import streamlit as st
import pandas as pd
import os
import json
import tempfile
from llama_parse import LlamaParse
from openai import OpenAI

# Header Streamlit
st.set_page_config(page_title="Parser PDF Piani Alimentari", layout="wide")
st.title("Analizzatore di Piani Alimentari da PDF")
st.write(
    "Carica un file PDF contenente un piano alimentare. L'app estrarrà il contenuto, "
    "lo strutturerà in formato JSON e poi lo visualizzerà come testo formattato."
)

# Secrets API Keys
try:
    LLAMA_CLOUD_API_KEY = st.secrets["LLAMA_CLOUD_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.sidebar.warning(
        "API keys non trovate in st.secrets. Inseriscile manualmente qui sotto per continuare."
    )
    LLAMA_CLOUD_API_KEY = st.sidebar.text_input("Llama Cloud API Key", type="password")
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")


# Funzioni per il parsing e l'estrazione
@st.cache_data(show_spinner=False)
def process_pdf_llamaparse(pdf_file_path):
    """
    Funzione per elaborare il PDF utilizzando LlamaParse.
    """
    if not LLAMA_CLOUD_API_KEY:
        st.error("API Key per Llama Cloud non fornita.")
        return None
    st.info("Avvio del parsing del PDF con LlamaParse...")
    try:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="it",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="anthropic-sonnet-3.7",
        )

        documents = parser.load_data(pdf_file_path)
        if documents:
            parsed_content = ""
            for doc in documents:
                parsed_content += doc.text + "\n"
            st.success("Parsing del PDF completato con successo.")
            return parsed_content
        else:
            st.error("Errore: Nessun contenuto parsato restituito da LlamaParse.")
            return None
    except Exception as e:
        st.error(f"Si è verificato un errore durante il parsing con LlamaParse: {e}")
        return None


def process_md_gpt(markdown_content):
    if not OPENAI_API_KEY:
        st.error("API Key per OpenAI non fornita.")
        return None
    if not markdown_content:
        st.warning("Nessun contenuto markdown da processare.")
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    json_schema = {
        "giorni": [
            {
                "giorno": "string",
                "colazione": {
                    "principale": [{"alimento": "string", "quantita": "string"}],
                    "alternative": [{"alimento": "string", "quantita": "string"}],
                },
                "spuntino_mattina": {
                    "principale": [{"alimento": "string", "quantita": "string"}],
                    "alternative": [{"alimento": "string", "quantita": "string"}],
                },
                "pranzo": {
                    "principale": [{"alimento": "string", "quantita": "string"}],
                    "alternative": [{"alimento": "string", "quantita": "string"}],
                },
                "spuntino_pomeriggio": {
                    "principale": [{"alimento": "string", "quantita": "string"}],
                    "alternative": [{"alimento": "string", "quantita": "string"}],
                },
                "cena": {
                    "principale": [{"alimento": "string", "quantita": "string"}],
                    "alternative": [{"alimento": "string", "quantita": "string"}],
                },
            }
        ],
        "note/consigli": ["string"],
    }

    prompt = f"""
    Analizza il seguente piano alimentare in formato markdown ed estrai le informazioni in formato JSON strutturato.

    CONTENUTO DEL PIANO ALIMENTARE:
    {markdown_content}

    ISTRUZIONI:
    1. Estrai i pasti per ogni giorno della settimana (lunedì-domenica)
    2. Per ogni pasto, identifica:
       - Alimenti principali con le loro quantità
       - Alternative (quando presenti) con le loro quantità
    3. Estrai i consigli/note generici del nutrizionista
    4. Se un pasto è indicato come "PASTO LIBERO", includi questa informazione come alimento principale
    5. Se il piano alimentare descrive una struttura di pasti generica, questa deve essere replicata per ogni giorno della settimana
    6. Se per un pasto (es. spuntino_mattina) non ci sono informazioni nel documento, lascia le liste "principale" e "alternative" vuote per quel pasto. Non omettere la chiave del pasto.
    7. Assicurati che ogni giorno da lunedì a domenica sia presente nell'output JSON. Se il documento non specifica pasti per un giorno, quel giorno dovrebbe comunque apparire con i campi dei pasti vuoti o con indicazioni di riposo/libero se presenti.

    FORMATO JSON RICHIESTO:
    {json.dumps(json_schema, indent=2)}

    IMPORTANTE:
    - Restituisci **esattamente** il JSON, senza testo introduttivo o conclusivo.
    - NON includere alcun commento, spiegazione o intestazione. Solo JSON valido.
    """

    model = "gpt-4o-mini"
    st.info(f"Processing del markdown con {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Sei un assistente specializzato nell'estrazione strutturata di dati da piani alimentari. Restituisci sempre e solo JSON valido secondo lo schema fornito.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )
        json_response_content = response.choices[0].message.content
        meal_plan_json = json.loads(json_response_content)
        st.success("Estrazione JSON completata con successo.")
        return meal_plan_json

    except json.JSONDecodeError as e:
        st.error(f"Errore nel parsing del JSON dalla risposta di GPT: {e}")
        st.text("Risposta ricevuta (potrebbe essere troncata):")
        st.code(
            json_response_content
            if "json_response_content" in locals()
            else "Nessuna risposta per JSONDecodeError"
        )
        return None
    except Exception as e:
        st.error(f"Errore durante la chiamata a {model}: {e}")
        return None


def json_to_text(json_data):
    if not json_data or "giorni" not in json_data:
        return "Dati JSON non validi o mancanti."

    output = []
    for giorno_data in json_data.get("giorni", []):
        output.append(f"Giorno: {giorno_data.get('giorno', 'Non specificato')}")
        for pasto_key in [
            "colazione",
            "spuntino_mattina",
            "pranzo",
            "spuntino_pomeriggio",
            "cena",
        ]:
            output.append(f"\n{pasto_key.replace('_', ' ').capitalize()}:")
            pasto_data = giorno_data.get(pasto_key, {})
            principale = pasto_data.get("principale", [])
            alternative = pasto_data.get("alternative", [])

            if principale:
                output.append("  - Principale:")
                for item in principale:
                    output.append(
                        f"    • {item.get('alimento', 'N/A')}: {item.get('quantita', 'N/A')}"
                    )
            else:
                output.append("  - Principale: Nessun dato")

            if alternative:
                output.append("  - Alternative:")
                for item in alternative:
                    output.append(
                        f"    • {item.get('alimento', 'N/A')}: {item.get('quantita', 'N/A')}"
                    )

        output.append("\n" + "-" * 30 + "\n")

    if "note/consigli" in json_data and json_data["note/consigli"]:
        output.append("Note/Consigli:")
        for nota in json_data["note/consigli"]:
            output.append(f"• {nota}")

    return "\n".join(output)


# Funzioni per creare i DataFrame
def get_weekly_plan(json_output):
    giorni = json_output.get("giorni", [])
    rows = []

    for giorno in giorni:
        giorno_row = {"giorno": giorno.get("giorno", "")}

        for pasto_key in [
            "colazione",
            "spuntino_mattina",
            "pranzo",
            "spuntino_pomeriggio",
            "cena",
        ]:
            pasto = giorno.get(pasto_key, {})
            contenuto = []
            for item in pasto.get("principale", []):
                alimento = item.get("alimento", "")
                quantita = item.get("quantita", "")
                contenuto.append(f"{alimento} ({quantita})")
            if pasto.get("alternative"):
                alternative = [
                    f"{item.get('alimento', '')} ({item.get('quantita', '')})"
                    for item in pasto["alternative"]
                ]
                contenuto.append("Alternative: " + ", ".join(alternative))

            giorno_row[pasto_key] = "\n".join(contenuto) if contenuto else ""

        rows.append(giorno_row)

    return pd.DataFrame(rows)


def get_notes(json_output):
    consigli = json_output.get("note/consigli", [])
    return pd.DataFrame({"note/consigli": consigli})


# Funzione per pulire la cache
def clear_cache():
    if "markdown_content" in st.session_state:
        del st.session_state.markdown_content
    if "meal_plan_json" in st.session_state:
        del st.session_state.meal_plan_json
    st.session_state.uploaded_file = None
    st.cache_data.clear()
    st.experimental_rerun()


# Pulsante per pulire la cache
st.sidebar.button("Svuota cache e carica nuovo PDF", on_click=clear_cache)

# Componente per il caricamento del file PDF
uploaded_file = st.file_uploader(
    "Carica il tuo file PDF", type="pdf", key="uploaded_file"
)

if uploaded_file is not None:
    if not LLAMA_CLOUD_API_KEY or not OPENAI_API_KEY:
        st.error("Per favore, inserisci le API key nella sidebar per procedere.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_file_path_for_parser = tmp_file.name

        st.markdown("---")
        st.subheader("1. Parsing del PDF")
        with st.spinner("Attendere prego: parsing in corso..."):
            if "markdown_content" not in st.session_state:
                st.session_state.markdown_content = process_pdf_llamaparse(
                    pdf_file_path_for_parser
                )
            markdown_content = st.session_state.markdown_content

        if os.path.exists(pdf_file_path_for_parser):
            os.remove(pdf_file_path_for_parser)

        if markdown_content:
            with st.expander(
                "Visualizza contenuto Markdown estratto (grezzo)", expanded=False
            ):
                st.markdown(f"```markdown\n{markdown_content}\n```")

            st.markdown("---")
            st.subheader("2. Estrazione strutturata del Piano Alimentare")
            with st.spinner("Attendere prego: estrazione in corso..."):
                if "meal_plan_json" not in st.session_state:
                    st.session_state.meal_plan_json = process_md_gpt(markdown_content)
                meal_plan_json = st.session_state.meal_plan_json

            if meal_plan_json:
                # with st.expander("Visualizza JSON Strutturato", expanded=False):
                #    st.json(meal_plan_json)

                st.subheader("2.1. Piano Alimentare Settimanale")
                weekly_plan_df = get_weekly_plan(meal_plan_json)
                st.dataframe(weekly_plan_df, use_container_width=True)

                # Visualizzazione dataframe note/consigli
                st.subheader("2.2. Note e Consigli")
                notes_df = get_notes(meal_plan_json)
                st.dataframe(notes_df, use_container_width=True)
                st.markdown(
                    "Puoi copiare i dati delle tabelle in Excel o Google Sheets."
                )

                st.markdown("---")
                st.subheader("3. Estrazione testuale del Piano Alimentare")
                json_text_output = json_to_text(meal_plan_json)

                st.text_area("Testo del Piano Alimentare", json_text_output, height=400)
                st.info(
                    "Per copiare il testo, selezionalo e usa Ctrl+C (o Cmd+C su Mac)."
                )

            else:
                st.error(
                    "Non è stato possibile generare il JSON strutturato dal markdown."
                )
        else:
            st.error("Non è stato possibile parsare il PDF.")
else:
    st.info("In attesa del caricamento di un file PDF.")

st.sidebar.markdown("---")
st.sidebar.header("Informazioni")
st.sidebar.info(
    "Questa app utilizza LlamaParse per estrarre il testo da PDF e un modello OpenAI GPT "
    "per strutturare le informazioni di un piano alimentare."
)
st.sidebar.info(
    "Assicurati di avere le API key valide e che i rispettivi servizi siano accessibili."
)
