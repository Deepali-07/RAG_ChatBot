"""
rag_engine.py — Core RAG logic
Stack: LangChain · ChromaDB · HuggingFace Embeddings · Groq LLM
"""

import os
import tempfile
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document


# ── Demo astrophysics knowledge base ──────────────────────────────────────────
DEMO_ASTROPHYSICS_TEXT = """
## Stellar Evolution and the Hertzsprung-Russell Diagram

Stars are born inside giant molecular clouds, regions of gas and dust that collapse under gravity. 
When the core temperature reaches about 10 million Kelvin, hydrogen fusion begins and a star is 
"born" on the main sequence of the Hertzsprung-Russell (HR) diagram. The HR diagram plots stellar 
luminosity against surface temperature (or spectral class). Main sequence stars fuse hydrogen in their 
cores; this phase lasts millions to billions of years depending on mass.

Low-mass stars (< 0.8 solar masses) remain on the main sequence for longer than the current age of 
the universe. Intermediate-mass stars like our Sun spend about 10 billion years on the main sequence. 
High-mass stars (> 8 solar masses) exhaust hydrogen in millions of years and end their lives as 
supernovae.

The Chandrasekhar limit is the maximum mass of a stable white dwarf star, approximately 1.4 solar 
masses. Above this limit, electron degeneracy pressure cannot prevent gravitational collapse. 
Subrahmanyan Chandrasekhar derived this limit in 1930, a discovery that later earned him the Nobel 
Prize in Physics (1983).

## Black Holes and Event Horizons

A black hole is a region of spacetime where gravity is so strong that nothing — not even light — 
can escape once it crosses the event horizon. The Schwarzschild radius defines the event horizon for 
a non-rotating black hole: r_s = 2GM/c², where G is the gravitational constant, M is the mass, and 
c is the speed of light.

Types of black holes:
- Stellar black holes: formed by gravitational collapse of massive stars (3–100 solar masses)
- Intermediate black holes: 100–100,000 solar masses, formation mechanisms debated
- Supermassive black holes: millions to billions of solar masses, found at galaxy centers
- Primordial black holes: hypothetical, formed in the early universe

Hawking radiation, predicted by Stephen Hawking in 1974, is thermal radiation theoretically emitted 
by black holes due to quantum effects near the event horizon. As a black hole emits Hawking radiation, 
it gradually loses mass in a process called black hole evaporation.

## Dark Matter and Dark Energy

Dark matter is a form of matter that does not interact with the electromagnetic force but whose 
presence can be inferred from gravitational effects on visible matter. Dark matter accounts for 
approximately 27% of the universe's total mass-energy content. Evidence includes galactic rotation 
curves (Vera Rubin, 1970s), gravitational lensing, and the cosmic microwave background.

Dark energy is the unknown form of energy that is accelerating the expansion of the universe. It 
accounts for approximately 68% of the total energy in the observable universe. The cosmological 
constant (Λ), introduced by Einstein and later revived, is the simplest model for dark energy.

The current cosmological model, ΛCDM (Lambda Cold Dark Matter), describes a universe composed of:
- ~5% ordinary (baryonic) matter
- ~27% cold dark matter
- ~68% dark energy

## Cosmic Microwave Background (CMB)

The CMB is thermal radiation filling the universe, a relic from the early universe when it became 
transparent to radiation, about 380,000 years after the Big Bang (recombination epoch). The CMB 
temperature is 2.725 K with tiny fluctuations (~1 part in 100,000). These anisotropies were first 
detected by COBE (1992), then mapped in detail by WMAP (2001) and Planck (2009–2013).

## Gravitational Waves

Gravitational waves are ripples in spacetime caused by accelerating massive objects. Predicted by 
Einstein's General Relativity (1915), they were first directly detected on September 14, 2015 by 
LIGO (Laser Interferometer Gravitational-Wave Observatory). The source was a binary black hole merger, 
1.3 billion light-years away. The detection was announced in February 2016 and led to the 2017 Nobel 
Prize in Physics for Weiss, Barish, and Thorne.

## The Big Bang and Cosmic Inflation

The Big Bang theory describes the origin of the universe from a hot, dense state approximately 
13.8 billion years ago. Cosmic inflation, proposed by Alan Guth in 1980, posits that the universe 
underwent exponential expansion in the first 10^-36 seconds, explaining the flatness and horizon 
problems. The observable universe is approximately 93 billion light-years in diameter.

## Neutron Stars and Pulsars

Neutron stars are the collapsed cores of massive stars (8–20 solar masses) after supernova explosions. 
They are incredibly dense: a teaspoon of neutron star material weighs about 10 million tons. 
A neutron star with a mass of ~1.4 solar masses has a radius of only ~10 km. 

Pulsars are rapidly rotating neutron stars emitting beams of electromagnetic radiation. The fastest 
known pulsar (PSR J1748-2446ad) rotates 716 times per second. Pulsars are used as cosmic clocks 
and have been used to indirectly confirm gravitational wave emission (Hulse-Taylor pulsar, Nobel 1993).

## Exoplanets and the Search for Life

As of 2024, over 5,500 confirmed exoplanets have been discovered. Detection methods include:
- Transit photometry: measuring brightness dips as planets cross their star (Kepler, TESS)
- Radial velocity: measuring Doppler shifts due to planetary gravitational pull
- Direct imaging: photographing planets directly (limited to large, young, distant planets)

The habitable zone (Goldilocks zone) is the range of orbital distances where liquid water could 
exist on a planet's surface. The James Webb Space Telescope (launched Dec 2021) is revolutionizing 
exoplanet atmosphere characterization through transmission spectroscopy.
"""


class RAGEngine:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key

        # HuggingFace embeddings — free, no key needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

        self.vectorstore = None
        self.qa_chain = None

        # Groq LLM (free tier available)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024,
            groq_api_key=groq_api_key,
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are CosmicRAG, an intelligent document assistant specialized in answering 
questions accurately based on provided document context. 

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the context provided above
- Be concise but thorough
- If the answer is not in the context, say "I don't have information about this in the loaded documents"
- Use clear formatting with bullet points or numbered lists when appropriate
- Cite specific details from the documents

ANSWER:"""
        )

    def load_pdfs(self, uploaded_files) -> Tuple[List[Document], int]:
        """Load and split uploaded PDF files."""
        all_docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Tag metadata
            for page in pages:
                page.metadata["source"] = file.name
                page.metadata["type"] = "uploaded_pdf"

            chunks = self.text_splitter.split_documents(pages)
            all_docs.extend(chunks)
            os.unlink(tmp_path)

        return all_docs, len(all_docs)

    def load_demo_astrophysics(self) -> Tuple[List[Document], int]:
        """Load built-in astrophysics knowledge base."""
        sections = DEMO_ASTROPHYSICS_TEXT.strip().split("\n\n## ")
        docs = []

        for i, section in enumerate(sections):
            if section.startswith("## "):
                section = section[3:]

            lines = section.split("\n", 1)
            title = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""

            if content:
                chunks = self.text_splitter.split_text(content)
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": f"Astrophysics KB · {title}",
                            "section": title,
                            "type": "demo",
                            "chunk": j,
                        }
                    )
                    docs.append(doc)

        return docs, len(docs)

    def build_vectorstore(self, documents: List[Document]) -> int:
        """Build ChromaDB vectorstore from documents."""
        if not documents:
            raise ValueError("No documents provided.")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="cosmicrag_kb",
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",          # Maximal Marginal Relevance for diversity
                search_kwargs={"k": 5, "fetch_k": 10},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template},
        )

        return len(documents)

    def query(self, question: str) -> Tuple[str, List[str]]:
        """Run RAG query and return answer + source citations."""
        if not self.qa_chain:
            raise ValueError("Knowledge base not built. Please upload documents first.")

        result = self.qa_chain.invoke({"query": question})

        answer = result["result"].strip()

        # Deduplicate sources
        seen = set()
        sources = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            label = f"{src}" + (f" · p.{page+1}" if page is not None else "")
            if label not in seen:
                seen.add(label)
                sources.append(label)

        return answer, sources[:4]  # max 4 source pills