const asyncHandler = require("express-async-handler");

const { MongoClient, ObjectId } = require('mongodb');
const AWS = require('aws-sdk');
const dotenv = require('dotenv');
const Joi = require('joi');
const { OpenAI } = require('openai');

const { LRUCache } = require('lru-cache');

// Load environment variables
dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Configure logging
const logger = {
  info: (msg) => console.log(msg),
  error: (msg) => console.error(msg)
};



// Global Bedrock client
let _bedrockClient = null;

// Environment variables with defaults
const MONGO_URI = process.env.MONGO_URI;
const MONGO_DB = process.env.MONGO_DB || 'medical_research';
const MONGO_COLLECTION = process.env.MONGO_COLLECTION || 'pubmed_articles';
const VECTOR_SEARCH_INDEX = process.env.VECTOR_SEARCH_INDEX || 'pubmed_vector_index';
const AWS_REGION = process.env.AWS_REGION || 'ap-south-1';

// Document cache
const docCache = new LRUCache({
  max: 1000,
  ttl: 1000 * 60 * 60, // 1 hour
});
// MongoDB connection
let dbClient = null;
async function connectDB() {
  if (!MONGO_URI) {
    throw new Error('MongoDB connection string not set');
  }
  
  if (!dbClient) {
    dbClient = await MongoClient.connect(MONGO_URI);
  }
  
  return dbClient.db(MONGO_DB);
}

// Schema validation
const vectorSearchSchema = Joi.object({
  query: Joi.string().required(),
  top_k: Joi.number().integer().min(1).max(100).default(5),
  min_score: Joi.number().optional(),
  filters: Joi.object().optional(),
  include_embeddings: Joi.boolean().default(false)
});

const ragQuerySchema = Joi.object({
  query: Joi.string().required(),
  top_k: Joi.number().integer().min(1).max(20).default(5),
  filters: Joi.object().optional(),
  temperature: Joi.number().min(0).max(1).default(0.0),
  max_tokens: Joi.number().integer().min(1).max(4000).default(1500),
  include_documents: Joi.boolean().default(true),
  model: Joi.string().default('anthropic.claude-3-opus-20240229-v1:0')
});

const clinicalTrialQuerySchema = Joi.object({
  query: Joi.string().required(),
  top_k: Joi.number().integer().min(1).max(30).default(8),
  phase: Joi.string().optional(),
  status: Joi.string().optional(),
  condition: Joi.string().optional(),
  intervention_type: Joi.string().optional(),
  max_tokens: Joi.number().integer().min(1).max(4000).default(2000),
  temperature: Joi.number().min(0).max(1).default(0.1),
  include_documents: Joi.boolean().default(true)
});

// Query classification constants
const DRUG_INTERACTION_KEYWORDS = new Set([
  "drug interaction", "drug-drug", "interaction", "contraindication", 
  "concomitant", "co-administration", "drug combination", "polypharmacy", 
  "medication interaction"
]);

const TREATMENT_KEYWORDS = new Set([
  "treatment", "therapy", "manage", "care", "intervention", "protocol", 
  "regimen", "guideline", "approach", "strategy"
]);

const DISEASE_KEYWORDS = new Set([
  "disease", "condition", "disorder", "syndrome", "pathology", "illness", 
  "symptoms", "diagnosis", "etiology"
]);

const PUBLIC_HEALTH_KEYWORDS = new Set([
  "public health", "policy", "population", "community", "prevention", 
  "screening", "surveillance", "outbreak", "epidemic", "pandemic"
]);

// Pre-defined system prompts
const SYSTEM_PROMPT = `You are HospiAgent's Medical Search Agent for Indian healthcare professionals. Answer directly without mentioning your role or how you formed your answer.

IMPORTANT GUIDELINES:
1. Use the provided medical literature as your primary source, but you may draw on prior knowledge to enhance and complement the evidence.
2. Focus on practical, actionable information for Indian healthcare professionals.
3. Consider the Indian context: local disease prevalence, available medications, treatment guidelines, healthcare resources, and cultural factors.

4. For drug interaction queries, structure your response EXACTLY as follows:
   **DRUG INTERACTION SUMMARY:**
   * **Severity:** [Critical/Major/Moderate/Minor] - Be very explicit about interaction severity
   * **Mechanism:** Clear, concise explanation of how the drugs interact
   * **Clinical Effects:** Bullet-point list of specific adverse effects or consequences
   * **Management Options:**
     - Specific dosage adjustments
     - Alternative medications
     - Monitoring parameters with specific thresholds
   * **Special Populations:** Considerations for pediatric, geriatric, pregnancy, renal/hepatic impairment
   * **Indian Context:** Availability of alternatives, local guidelines, cost considerations

5. For treatment or diagnostic queries, structure your response as follows:
   **CLINICAL ANSWER:**
   * **Key Recommendation:** 1-2 sentence direct answer
   * **Evidence Summary:** 
     - Bullet-point list of main findings from literature
     - Include efficacy rates, confidence intervals, or p-values when available
   * **Treatment Algorithm:** Step-by-step approach in numbered list format
   * **Monitoring:** Specific parameters to track with frequency and thresholds
   * **Indian Context:** 
     - Availability of treatments/diagnostics
     - Cost considerations
     - Local guidelines
   * **Red Flags:** Warning signs requiring urgent attention

6. For disease/condition queries, structure your response as follows:
   **CONDITION OVERVIEW:**
   * **Definition:** Concise clinical definition
   * **Epidemiology:** Key statistics, especially for Indian population
   * **Clinical Presentation:** Bullet-point list of symptoms and signs by frequency
   * **Diagnostic Approach:** Clear stepwise process
   * **Management:** First-line, second-line options in structured format
   * **Prevention:** Evidence-based preventive measures
   * **Special Considerations for India:** Regional variations, resource constraints

7. For public health or policy queries:
   **PUBLIC HEALTH PERSPECTIVE:**
   * **Current Status:** Data-driven summary of the situation in India
   * **Key Challenges:** Bullet-point list of barriers
   * **Evidence-Based Interventions:** Prioritized list with efficacy data
   * **Resource Optimization:** How to implement with limited resources
   * **Metrics & Evaluation:** How to measure success
   * **Policy Recommendations:** Clear, actionable steps

DO NOT cite document numbers, mention insufficient context, or include meta-commentary about your answer. Start directly with the content in the required format.

ALWAYS use bullet points, numbered lists, bold headers, and clear section breaks to make information scannable and actionable.`;

const CLINICAL_TRIAL_SYSTEM_PROMPT = `You are HospiAgent's Clinical Trial Search Agent for Indian healthcare professionals. Answer directly without mentioning your role or how you formed your answer.

IMPORTANT GUIDELINES:
1. Use the provided clinical trial literature as your primary source, but you may draw on prior knowledge to enhance and complement the evidence.
2. Focus on practical, actionable information about clinical trials for Indian healthcare professionals.
3. Consider the Indian context: local regulatory environment, standard of care, available treatments, and cultural factors.

Structure your response with the following sections:

**CLINICAL TRIAL SUMMARY:**
* **Overview:** Brief summary of relevant trials addressing the query
* **Study Design:** Details on trial design, phases, and methodology
* **Inclusion/Exclusion Criteria:** Key eligibility criteria for the trials
* **Interventions:** Description of treatments or interventions being studied
* **Outcomes:** Primary and secondary endpoints with available results if completed
* **Safety Profile:** Notable adverse events and safety considerations
* **Indian Context:** 
  - Relevance to Indian patient populations
  - Whether trials are/were conducted in India
  - Implications for Indian healthcare practices
  - Regulatory status in India if applicable
* **Clinical Implications:** How findings might impact current clinical practice
* **Limitations:** Important caveats or limitations of the trial data

DO NOT cite document numbers or mention insufficient context. Format your answer with bullet points, numbered lists, bold headers, and clear section breaks to make information scannable for busy clinicians.`;

// Context processor class
class ContextProcessor {
  constructor(maxContextLength = 15000) {
    this.maxContextLength = maxContextLength;
  }

  formatDocument(doc, idx, score) {
    // Format document for both context and result output
    const searchResult = {
      id: doc._id.toString(),
      title: doc.Title || '',
      authors: doc.Authors || '',
      abstract: doc.Abstract || '',
      publication_date: doc['Publication Date'] ? doc['Publication Date'].toString() : '',
      journal: doc.Journal || '',
      doi: doc.DOI || '',
      doi_link: doc['DOI Link'] || '',
      link: doc.Link || '',
      score: score
    };

    // Create context chunk efficiently with array join
    const contextParts = [
      `DOCUMENT ${idx+1} [Score: ${score.toFixed(4)}]`,
      `Title: ${doc.Title || 'No title'}`
    ];

    if (doc.Authors) {
      contextParts.push(`Authors: ${doc.Authors}`);
    }

    if (doc.Journal) {
      contextParts.push(`Journal: ${doc.Journal}`);
    }

    if (doc['Publication Date']) {
      contextParts.push(`Date: ${doc['Publication Date']}`);
    }

    if (doc.DOI) {
      contextParts.push(`DOI: ${doc.DOI}`);
    }

    // Always include abstract for medical papers
    contextParts.push(`Abstract: ${doc.Abstract || 'No abstract'}`);

    // Add content using the most efficient available field
    let content = null;
    if (doc['Cleaned Text']) {
      content = doc['Cleaned Text'];
    } else if (doc['Full Text']) {
      content = doc['Full Text'];
    }

    if (content) {
      // Efficient text truncation
      if (content.length > 3000) {
        // Try to break at paragraph
        let breakPoint = content.lastIndexOf('\n\n', 3000);
        if (breakPoint === -1) {
          // Try to break at sentence
          breakPoint = content.lastIndexOf('. ', 3000);
          if (breakPoint !== -1) {
            breakPoint += 1; // Include the period
          }
        }
        if (breakPoint === -1) {
          breakPoint = 3000;
        }
        
        contextParts.push(`Content: ${content.substring(0, breakPoint)}...`);
      } else {
        contextParts.push(`Content: ${content}`);
      }
    }

    return [searchResult, contextParts.join('\n')];
  }

  async processSearchResults(results, db) {
    const formattedResults = [];
    const contextChunks = [];
    let totalContextLength = 0;

    const collection = db.collection(MONGO_COLLECTION);

    for (let idx = 0; idx < results.length; idx++) {
      const result = results[idx];
      const docId = result._id;
      const docIdStr = docId.toString();
      const score = result.score || 0.0;

      // Fetch document from cache or database
      let fullDoc;
      if (docCache.has(docIdStr)) {
        fullDoc = docCache.get(docIdStr);
      } else {
        try {
          fullDoc = await collection.findOne({ _id: docId });
          if (fullDoc) {
            docCache.set(docIdStr, fullDoc);
          }
        } catch (error) {
          logger.error(`Error fetching document ${docIdStr}: ${error.message}`);
          continue;
        }
      }

      if (fullDoc) {
        // Format document for both result and context
        const [searchResult, contextChunk] = this.formatDocument(fullDoc, idx, score);

        formattedResults.push(searchResult);

        // Check if adding this chunk would exceed the maximum context length
        const chunkLength = contextChunk.length;
        if (totalContextLength + chunkLength <= this.maxContextLength) {
          contextChunks.push(contextChunk);
          totalContextLength += chunkLength;
        } else {
          // If we're about to exceed the limit, only include the most relevant documents
          break;
        }
      }
    }

    return [formattedResults, contextChunks];
  }
}

// Helper function to execute search and process results
async function executeSearch(queryText, filters, topK, db) {
  // Build optimized search pipeline
  const searchPipeline = [
    {
      $search: {
        index: VECTOR_SEARCH_INDEX,
        text: {
          query: queryText,
          path: { wildcard: "*" }
        }
      }
    },
    {
      $project: {
        _id: 1, // Only project the ID and score
        score: { $meta: "searchScore" }
      }
    },
    { $limit: topK }
  ];

  // Add filters if provided
  if (filters && Object.keys(filters).length > 0) {
    // Insert a $match stage after the $search stage
    searchPipeline.splice(1, 0, { $match: filters });
  }

  // Execute the search pipeline
  const collection = db.collection(MONGO_COLLECTION);
  const results = await collection.aggregate(searchPipeline).toArray();

  // Process results
  const processor = new ContextProcessor();
  const [formattedResults, contextChunks] = await processor.processSearchResults(results, db);

  return [formattedResults, contextChunks];
}

// Helper function to determine query type and format hint
function getQueryTypeAndHint(queryText) {
  const queryLower = queryText.toLowerCase();
  const queryWords = new Set(queryLower.split(/\s+/));
  
  // Check for query types using set intersection
  if ([...DRUG_INTERACTION_KEYWORDS].some(keyword => queryLower.includes(keyword))) {
    return "This appears to be a drug interaction query. Follow the DRUG INTERACTION SUMMARY format with all sections.";
  } else if ([...TREATMENT_KEYWORDS].some(keyword => queryWords.has(keyword))) {
    return "This appears to be a treatment query. Follow the CLINICAL ANSWER format with all sections.";
  } else if ([...DISEASE_KEYWORDS].some(keyword => queryWords.has(keyword))) {
    return "This appears to be a disease/condition query. Follow the CONDITION OVERVIEW format with all sections.";
  } else if ([...PUBLIC_HEALTH_KEYWORDS].some(keyword => queryLower.includes(keyword))) {
    return "This appears to be a public health query. Follow the PUBLIC HEALTH PERSPECTIVE format with all sections.";
  }
  
  return "Structure your response with clear headers, bullet points, and numbered lists for each major section.";
}

// Helper function to determine clinical trial format hint
function getClinicalTrialHint(queryText) {
  const queryLower = queryText.toLowerCase();
  
  if (/safety|adverse|side effect|toxicity/.test(queryLower)) {
    return "This appears to be a safety query. Focus on the Safety Profile section with detailed adverse event data.";
  } else if (/efficacy|effectiveness|outcome|result/.test(queryLower)) {
    return "This appears to be an efficacy query. Focus on the Outcomes section with detailed efficacy data.";
  } else if (/design|methodology|protocol|inclusion|exclusion/.test(queryLower)) {
    return "This appears to be a study design query. Focus on the Study Design and Inclusion/Exclusion Criteria sections.";
  } else if (/india|indian|local|regional/.test(queryLower)) {
    return "This appears to be a query about Indian context. Focus on the Indian Context section and regional relevance.";
  }
  
  return "Provide a balanced summary across all sections, highlighting the most relevant clinical trial information.";
}

// Build clinical trial specific filters
function buildClinicalTrialFilters(query) {
  // Start with base filters that identify clinical trial documents
  const filters = {
    $or: [
      { Title: { $regex: "clinical trial|trial|phase|randomized|randomised", $options: "i" } },
      { Abstract: { $regex: "clinical trial|phase [I|II|III|IV]|randomized|randomised|NCT[0-9]", $options: "i" } },
      { Keywords: { $regex: "clinical trial", $options: "i" } },
      { "Document Type": { $regex: "clinical trial", $options: "i" } }
    ]
  };
  
  // Add user-specified filters
  const andConditions = [];
  
  if (query.phase) {
    const phaseRegex = `phase ${query.phase}|phase-${query.phase}|phase${query.phase}`;
    andConditions.push({
      $or: [
        { Title: { $regex: phaseRegex, $options: "i" } },
        { Abstract: { $regex: phaseRegex, $options: "i" } }
      ]
    });
  }
  
  if (query.status) {
    const statusRegex = query.status;
    andConditions.push({
      $or: [
        { Title: { $regex: statusRegex, $options: "i" } },
        { Abstract: { $regex: statusRegex, $options: "i" } }
      ]
    });
  }
  
  if (query.condition) {
    const conditionRegex = query.condition;
    andConditions.push({
      $or: [
        { Title: { $regex: conditionRegex, $options: "i" } },
        { Abstract: { $regex: conditionRegex, $options: "i" } },
        { Keywords: { $regex: conditionRegex, $options: "i" } }
      ]
    });
  }
  
  if (query.intervention_type) {
    const interventionRegex = query.intervention_type;
    andConditions.push({
      $or: [
        { Title: { $regex: interventionRegex, $options: "i" } },
        { Abstract: { $regex: interventionRegex, $options: "i" } }
      ]
    });
  }
  
  // If the user provided additional filters, merge them
  if (query.filters) {
    Object.entries(query.filters).forEach(([key, value]) => {
      andConditions.push({ [key]: value });
    });
  }
  
  // Add $and conditions if any were created
  if (andConditions.length > 0) {
    filters.$and = andConditions;
  }
  
  return filters;
}

// RAG endpoint

const ragQuery = (async  (req,res) => {

  const startTime = Date.now();
  
  try {
    // Validate request
    const { error, value: query } = ragQuerySchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    // Connect to database
    const db = await connectDB();
    
    // Execute search
    const [formattedResults, contextChunks] = await executeSearch(
      query.query,
      query.filters,
      query.top_k,
      db
    );
    
    // Check if we have any results
    if (!contextChunks || contextChunks.length === 0) {
      return res.json({
        query: query.query,
        answer: "No relevant medical documents were found for your query.",
        documents: query.include_documents ? [] : undefined,
        execution_time_ms: Date.now() - startTime
      });
    }
    
    // Combine context chunks
    const context = "\n" + "=".repeat(50) + "\n" + contextChunks.join("\n") + "\n" + "=".repeat(50) + "\n";
    
    try {
      const formatHint = getQueryTypeAndHint(query.query);
      
      // Create user prompt
      const userPrompt = `QUESTION: ${query.query}

Please use the following documents as your primary source of information:

${context}

${formatHint}

DO NOT include any introductory statements about synthesizing information or using documents. DO NOT cite document numbers. Start directly with your answer in the required format. If the documents don't contain all the necessary information, use your medical knowledge to provide a complete answer without mentioning gaps in the provided context. Use formatting extensively to make your answer scannable.`;
      
      
      
    
      const completion = await openai.chat.completions.create({
        model: "gpt-4o-mini",  // You can choose any OpenAI model available to you
        max_tokens: query.max_tokens,
        temperature: query.temperature,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userPrompt }
        ],
      });
      
     
      const answer = completion.choices[0].message.content;
      
      // Return response
      return res.json({
        query: query.query,
        answer: answer,
        documents: query.include_documents ? formattedResults : undefined,
        execution_time_ms: Date.now() - startTime
      });
      
    } catch (error) {
      logger.error(`Error generating answer with Bedrock Claude: ${error.message}`);
      return res.status(500).json({
        query: query.query,
        answer: `Error generating answer with Bedrock Claude: ${error.message}`,
        documents: query.include_documents ? formattedResults : undefined,
        execution_time_ms: Date.now() - startTime
      });
    }
    
  } catch (error) {
    logger.error(`Error in RAG: ${error.message}`);
    return res.status(500).json({ error: `RAG failed: ${error.message}` });
  }
});

// Clinical trial RAG endpoint
 const getClinicalTrials = asyncHandler (async (req,res) => {
  const startTime = Date.now();
  
  try {
    // Validate request
    const { error, value: query } = clinicalTrialQuerySchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }
    
    // Connect to database
    const db = await connectDB();
    
    // Build clinical trial specific filters
    const filters = buildClinicalTrialFilters(query);
    
    // Execute search
    const [formattedResults, contextChunks] = await executeSearch(
      query.query,
      filters,
      query.top_k,
      db
    );
    
    // Check if we have any results
    if (!contextChunks || contextChunks.length === 0) {
      return res.json({
        query: query.query,
        answer: "No relevant clinical trial documents were found for your query. Consider broadening your search terms or checking for alternative terminology.",
        documents: query.include_documents ? [] : undefined,
        execution_time_ms: Date.now() - startTime
      });
    }
    
    // Combine context chunks
    const context = "\n" + "=".repeat(50) + "\n" + contextChunks.join("\n") + "\n" + "=".repeat(50) + "\n";
    
    try {
      
      // Get clinical trial specific format hint
      const formatHint = getClinicalTrialHint(query.query);
      
      // Create user prompt efficiently - tailored for clinical trials
      const userPrompt = `QUESTION ABOUT CLINICAL TRIALS: ${query.query}

Please use the following clinical trial documents as your primary source of information:

${context}

${formatHint}

DO NOT include any introductory statements about synthesizing information or using documents. DO NOT cite document numbers. Start directly with your answer using the CLINICAL TRIAL SUMMARY format. If the documents don't contain all the necessary information, use your medical knowledge to provide a complete answer without mentioning gaps in the provided context. Use formatting extensively to make your answer scannable.`;
      
     
     
      
      const completion = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        max_tokens: query.max_tokens,
        temperature: query.temperature,
        messages: [
          { role: "system", content: CLINICAL_TRIAL_SYSTEM_PROMPT },
          { role: "user", content: userPrompt }
        ],
      });
      
      const answer = completion.choices[0].message.content;
      
      // Return response
      return res.json({
        query: query.query,
        answer: answer,
        documents: query.include_documents ? formattedResults : undefined,
        execution_time_ms: Date.now() - startTime
      });
      
    } catch (error) {
      logger.error(`Error generating answer with Bedrock Claude for clinical trials: ${error.message}`);
      return res.status(500).json({
        query: query.query,
        answer: `Error generating answer about clinical trials: ${error.message}`,
        documents: query.include_documents ? formattedResults : undefined,
        execution_time_ms: Date.now() - startTime
      });
    }
    
  } catch (error) {
    logger.error(`Error in clinical trials RAG: ${error.message}`);
    return res.status(500).json({ error: `Clinical trials RAG failed: ${error.message}` });
  }
});



module.exports =  {
  ragQuery,
  getClinicalTrials
}