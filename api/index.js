
const express = require('express');
const bodyParser = require('body-parser');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { BraveSearch } = require('@langchain/community/tools/brave_search');
const OpenAI = require('openai');
const cheerio = require('cheerio');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(bodyParser.json());
const port = process.env.PORT;

/* Initialize Groq and embeddings */
let openai = new OpenAI({
    baseURL: 'https://api.groq.com/openai/v1',
    apiKey: process.env.GROQ_API_KEY,
});
const embeddings = new OpenAIEmbeddings();

app.get('/', async(req, res) => {
    res.status(200).json('A knockoff Perplexity API ;)');
})

app.post('/go', async (req, res) => {
    try {
        const {
            message,
            returnSources = true,
            returnFollowUpQuestions = true,
            embedSourcesInLLMResponse = false,
            textChunkSize = 800,
            textChunkOverlap = 200,
            numberOfSimilarityResults = 2,
            numberOfPagesToScan = 4
        } = req.body;
    
        async function rephraseInput(inputString) {
            const groqResponse = await openai.chat.completions.create({
                model: "mixtral-8x7b-32768",
                messages: [
                    { role: "system", content: `You are a rephraser and always respond with a rephrased version of the input 
                    that is given to a search engine API. Always be succint and use the same words as the input. 
                    ONLY RETURN THE REPHRASED VERSION OF THE INPUT.` },
                    { role: "user", content: inputString },
                ],
            });
            return groqResponse.choices[0].message.content;
        }
        
        async function searchEngineForSources(message) {
            const loader = new BraveSearch({ apiKey: process.env.BRAVE_SEARCH_API_KEY });
            const rephrasedMessage = await rephraseInput(message);
            const docs = await loader.call(rephrasedMessage, { count: numberOfPagesToScan });
            const normalizedData = normalizeData(docs);
            return await Promise.all(normalizedData.map(fetchAndProcess));
        }
        
        function normalizeData(docs) {
            return JSON.parse(docs)
            .filter((doc) => doc.title && doc.link && !doc.link.includes("brave.com"))
            .slice(0, numberOfPagesToScan)
            .map(({ title, link }) => ({ title, link }));
        }
        
        const fetchPageContent = async (link) => {
            try {
                const response = await fetch(link);
                if (!response.ok) 
                {
                    return "";
                }
                const text = await response.text();
                return extractMainContent(text, link);
            } catch (error) {
                console.error(`Error fetching page content for ${link}:`, error);
                return '';
            }
        };
        
        const fetchAndProcess = async (item) => {
            const htmlContent = await fetchPageContent(item.link);
            if (htmlContent && htmlContent.length < 250) return null;
            const splitText = await new RecursiveCharacterTextSplitter({ chunkSize: textChunkSize, chunkOverlap: textChunkOverlap }).splitText(htmlContent);
            const vectorStore = await MemoryVectorStore.fromTexts(splitText, { link: item.link, title: item.title }, embeddings);
            vectorCount++;
            return await vectorStore.similaritySearch(message, numberOfSimilarityResults);
        };
        
        function extractMainContent(html, link) {
            const $ = html.length ? cheerio.load(html) : null
            $("script, style, head, nav, footer, iframe, img").remove();
            return $("body").text().replace(/\s+/g, " ").trim();
        };
    
        let vectorCount = 0;
        const sources = await searchEngineForSources(message, textChunkSize, textChunkOverlap);
        const sourcesParsed = sources.map(group =>
            group.map(doc => {
                const title = doc.metadata.title;
                const link = doc.metadata.link;
                return { title, link };
            })
                .filter((doc, index, self) => self.findIndex(d => d.link === doc.link) === index)
        );
        const chatCompletion = await openai.chat.completions.create({
            messages:
                [{
                    role: "system", content: `
                - Here is my query "${message}", respond back with an answer that is as long as possible. If you can't find any relevant results, respond with "No relevant results found." 
                - ${embedSourcesInLLMResponse ? "Return the sources used in the response with iterable numbered markdown style annotations." : ""}" : ""}`
                },
                { role: "user", content: ` - Here are the top results from a similarity search: ${JSON.stringify(sources)}. ` },
                ], stream: true, model: "mixtral-8x7b-32768"
        });
        let responseTotal = "";
        for await (const chunk of chatCompletion) {
            if (chunk.choices[0].delta && chunk.choices[0].finish_reason !== "stop") 
            {
                process.stdout.write(chunk.choices[0].delta.content);
                responseTotal += chunk.choices[0].delta.content;
            } 
            else 
            {
                let responseObj = {};
                responseObj.userMessage = message;
                returnSources ? responseObj.sources = sourcesParsed : null;
                responseObj.answer = responseTotal;
                returnFollowUpQuestions ? responseObj.suggesstedQuestions = await generateFollowUpQuestions(responseTotal) : null;
                res.status(200).json(responseObj);
            }
        }
    } catch (error) {
        res.status(500).json({
            error: error.message,
        });
    }
});

async function generateFollowUpQuestions(responseText) {
    const groqResponse = await openai.chat.completions.create({
        model: "mixtral-8x7b-32768",
        messages: [
            { role: "system", content: `You are a question generator. Generate 3 follow-up questions 
            based on the provided text. Return the questions in an array format.` },
            {
                role: "user",
                content: `Generate 3 follow-up questions based on the following text:\n\n${responseText}\n\nReturn 
                the questions in the following format: ["Question 1", "Question 2", "Question 3"]`
            }
        ],
    });
    return JSON.parse(groqResponse.choices[0].message.content);
}

app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
})