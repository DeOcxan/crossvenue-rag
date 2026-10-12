# Domain research: what developers ask

The real developer questions and the near-duplicate traps of this corpus that the evaluation questions are drawn from ([methodology §2](methodology.md#2-synthetic-question-generation-used-as-a-drafting-aid-only), decision 4).

## Method

The themes come from GitHub search-API queries across `ccxt/ccxt`, `bybit-exchange/pybit`, `binance/binance-connector-python`, `hyperliquid-dex/hyperliquid-python-sdk`, `dydxprotocol/v4-clients`, `Uniswap/v3-sdk`, `Uniswap/sdks`, `Uniswap/docs` and the PancakeSwap org. They were cross-referenced against the FAQ, troubleshooting and error-code pages inside the corpus itself (the chunks whose path or heading matches `faq|troubleshoot|error|common|issue|problem`).

`pancakeswap/pancake-frontend` is not reachable through the GitHub API (repo GET returns 404, search returns 422), so PancakeSwap developer Q&A could not be mined at volume.

## Ranked themes

Ranked by how often each problem recurred in those searches, with corpus applicability.

### 1. Timestamp, clock skew and `recvWindow`

Signed requests rejected because the local clock drifted from exchange time.

| Venues | Why it discriminates retrieval |
|---|---|
| binance, bybit, bingx, ccxt; no `recvWindow` on dydx and hyperliquid | The exact numeric answer differs per venue and the shared phrasing is near-identical (trap T4). |

Issue titles:

- "Binance error: Timestamp for this request was 1000ms ahead of the server's time" ([ccxt#850](https://github.com/ccxt/ccxt/issues/850))
- "Binance: Timestamp for this request is outside of the recvWindow" ([ccxt#936](https://github.com/ccxt/ccxt/issues/936)); Bybit variant: "Local Time Out of Sync Causing Authentication Failure" ([pybit#300](https://github.com/bybit-exchange/pybit/issues/300))

### 2. Authentication and signature construction

Signature mismatch, `-2015 Invalid API-key, IP, or permissions`, IP allowlists, wrong key type, per-endpoint signing rules.

| Venues | Why it discriminates retrieval |
|---|---|
| binance (three key types: Ed25519/HMAC/RSA), bybit (`X-BAPI-SIGN`), bingx (`X-BX-APIKEY` + a mandatory `X-SOURCE-KEY` header), hyperliquid (EIP-712 agent wallet), dydx (Cosmos permissioned keys) | Binance's, Bybit's and BingX's docs all specify request signing with HMAC over SHA-256. Binance spells it `HMAC-SHA-256`, Bybit and BingX `HMAC-SHA256` and `HMAC_SHA256`, and BingX alone `HMAC SHA256`. Each describes a different canonical string: the corpus's largest "same words, different mechanism" family. |

Issue titles:

- "`-2015` Error Code: Invalid API-key, IP, or permissions for action." ([ccxt#25666](https://github.com/ccxt/ccxt/issues/25666))
- "[dYdX] Signature verification failed when placing order via CCXT" ([ccxt#27865](https://github.com/ccxt/ccxt/issues/27865))

### 3. Precision, tick size, lot size and minimum notional

Orders rejected by exchange filters because price or quantity was not rounded to the market's step.

| Venues | Why it discriminates retrieval |
|---|---|
| binance (`LOT_SIZE`/`MIN_NOTIONAL`/`PRICE_FILTER` filter objects), bybit (`lotSizeFilter.qtyStep`, `minNotionalValue`), bingx (`quantityPrecision` as an integer), hyperliquid (`szDecimals` and `nSigFigs`) | Four venues, four incompatible data models for one concept, expressed in overlapping words. |

Issue titles:

- "Binance open order return 400 Bad Request `{"code":-1013,"msg":"Filter failure: LOT_SIZE"}`" ([ccxt#663](https://github.com/ccxt/ccxt/issues/663))
<!-- verbatim GitHub issue title --><!-- vale crossvenue.EmDash = NO -->
- "Binance client throws MIN_NOTIONAL error when it's not supposed to" ([ccxt#1972](https://github.com/ccxt/ccxt/issues/1972)); Hyperliquid flavor: "Market order — ERROR: Order has invalid price" ([hyperliquid-python-sdk#132](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/132))
<!-- vale crossvenue.EmDash = YES -->

### 4. WebSocket subscription, reconnection and order-book resync

1006/1008 disconnects, silent stalls, no resubscribe after reconnect, ghost order-book levels, ping/pong timeouts, subscription caps.

| Venues | Why it discriminates retrieval |
|---|---|
| binance, bybit, bingx, hyperliquid and dydx, with five mutually incompatible keepalive protocols, and ccxt, which wraps them | "Who pings whom and how often" has five different correct answers and one shared vocabulary (T7). |

Issue titles:

- "watchOrderBookForSymbols missing updates" ([ccxt#23251](https://github.com/ccxt/ccxt/issues/23251))
- "WebsocketManager lacks reconnection and subscription restoration" ([hyperliquid-python-sdk#191](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/191))

### 5. Rate limits, request weight and IP bans

Hitting 429 / `-1003` / Bybit `10006`, not knowing about `enableRateLimit`, per-endpoint weight costs, IP-level vs account-level limits.

| Venues | Why it discriminates retrieval |
|---|---|
| binance, bybit, bingx, hyperliquid, dydx and uniswap. Chunks containing "rate limit" (`crossvenue count-term`): bingx 229, bybit 124, binance 42, ccxt 29, dydx 16, hyperliquid 14, uniswap 7, pancakeswap 0 | The same phrases ("too many requests", "rate limit exceeded") lead to different codes and ban lengths per venue (T5). |

Issue titles:

- "Polling Binance fetchTickers every 5 seconds triggers a ban" ([ccxt#1027](https://github.com/ccxt/ccxt/issues/1027))
- "Rate limit error. Response from bybit without X-Bapi-Limit-Reset-Timestamp" ([pybit#340](https://github.com/bybit-exchange/pybit/issues/340))

### 6. Order types, time-in-force and attached TP/SL

`postOnly`/`reduceOnly`/IOC/FOK semantics differ per venue; attaching a stop-loss or take-profit to an entry is a "how do I…" asked across the SDKs.

| Venues | Why it discriminates retrieval |
|---|---|
| binance, bybit, bingx, hyperliquid, dydx, ccxt | For time-in-force, one term names a different mechanism on each venue (T6). For TP/SL, a venue whose docs are thin makes a question risk becoming unanswerable by accident rather than by design. |

Issue titles:

- "fetchOpenOrder not work for stopLoss order" ([ccxt#24377](https://github.com/ccxt/ccxt/issues/24377))
- "How to set Take Profit & Stop Loss programmatically?" ([hyperliquid-python-sdk#280](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/280)); "Can't Set SL/TP With Spot Market Order" ([pybit#200](https://github.com/bybit-exchange/pybit/issues/200))

### 7. Spot vs derivatives vs unified account routing

`defaultType`, Bybit `category=linear|inverse|spot|option`, UTA vs classic; the wrong context silently returns empty balances rather than an error.

| Venues | Why it discriminates retrieval |
|---|---|
| bybit, binance, bingx, hyperliquid, ccxt | Binance here is spot only, the corpus's largest silent gap: Binance futures is absent from the corpus, and a question phrased for it can still be answerable through CCXT's documentation ([datasheet, known limitations](datasheet.md#known-limitations)). Such a question is material for the unanswerable category only when CCXT's documentation does not answer it either, and a risk for the fact category. |

Issue titles:

- "Bybit v5 not supporting normal accounts correctly" ([ccxt#19311](https://github.com/ccxt/ccxt/issues/19311))
- "ERROR:root:Exception while placing trade order: Illegal category (ErrCode: 10001)" ([ccxt#23902](https://github.com/ccxt/ccxt/issues/23902))

### 8. Leverage and margin mode (isolated vs cross)

`setLeverage`/`setMarginMode` silently no-op, must be set before the order, cannot change with an open position.

| Venues | Why it discriminates retrieval |
|---|---|
| bybit, bingx, hyperliquid, ccxt, dydx; not binance (spot-only) | The enum values differ for one concept: bingx uses `ISOLATED`/`CROSSED`, hyperliquid uses `strictIsolated`/`noCross`/`normal`. |

Issue titles:

- "how to set leverage on bybit?" ([ccxt#6919](https://github.com/ccxt/ccxt/issues/6919))
<!-- verbatim GitHub issue title --><!-- vale crossvenue.EmDash = NO -->
- "Kucoin Future — CROSS margin mode orders are not placed" ([ccxt#25592](https://github.com/ccxt/ccxt/issues/25592))
<!-- vale crossvenue.EmDash = YES -->

### 9. Pagination, `since`/`limit` and history depth

`since` ignored or mis-scaled, only the last N candles returned, no cursor loop, per-venue maximum windows.

| Venues | Why it discriminates retrieval |
|---|---|
| binance (offset + time window, 24-hour span caps), bybit (opaque `nextPageCursor`), bingx (`pageIndex`/`pageSize`, and units that change per endpoint), hyperliquid (hard 20-level cap on `l2Book`), dydx | Three incompatible pagination models share the word `limit` (T10). The shared vocabulary is one generic token rather than a whole paragraph. |

Issue titles:

<!-- verbatim GitHub issue title --><!-- vale crossvenue.EmDash = NO -->
- "Blofin — fetch_ohlcv is not working properly, especially 'since'" ([ccxt#24677](https://github.com/ccxt/ccxt/issues/24677))
<!-- vale crossvenue.EmDash = YES -->
- "Klines for '1d' interval returns only 3 values" ([binance-connector-python#224](https://github.com/binance/binance-connector-python/issues/224))

### 10. Testnet, sandbox and demo routing

`set_sandbox_mode` does not switch every endpoint; demo accounts need different hosts or key types; testnet data is malformed.

| Venues | Why it discriminates retrieval |
|---|---|
| the seven venues and ccxt (`sandbox`), each naming its test environment differently: binance has two (`testnet.binance.vision` and a separate Demo Mode at `demo-api.binance.com`), bingx calls it "Production Simulated" (`open-api-vst.bingx.com`) and never uses the word testnet as a hostname, hyperliquid requires flipping both the URL and a signed `hyperliquidChain` field, pancakeswap's "testnet" means BSC testnet contract addresses | Every venue names its test environment differently, and Binance's mainnet and testnet trees are the corpus's largest internal duplication (T8). |

Issue titles:

- "Binance Futures Testnet private API calls routed to Live Spot API (Invalid Api-Key ID)" ([ccxt#26487](https://github.com/ccxt/ccxt/issues/26487))
- "Demo API is not working" ([pybit#203](https://github.com/bybit-exchange/pybit/issues/203)); "testnet API returns invalid token indices causing IndexError" ([hyperliquid-python-sdk#275](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/275))

### 11. DEX: pool discovery, fee tier and tick data

"Which of the three fee-tier pools is the real price?", `computePoolAddress` returning the wrong address, "no tick data provider was given".

| Venues | Why it discriminates retrieval |
|---|---|
| uniswap, pancakeswap | Uniswap's medium tier is 0.30%, PancakeSwap's is 0.25%, and PancakeSwap on Solana has 18 tiers (T1): one question, a different number per venue, in shared vocabulary. |

Issue titles:

- "error: no tick data provider was given" ([v3-sdk#184](https://github.com/Uniswap/v3-sdk/issues/184))
<!-- verbatim GitHub issue title --><!-- vale crossvenue.USSpelling = NO -->
- "What is the right price amongst 3 Liquidity Pools of the same Token Pair on V3?" ([v3-sdk#53](https://github.com/Uniswap/v3-sdk/issues/53))
<!-- vale crossvenue.USSpelling = YES -->
- PancakeSwap's own developers ask the comparison directly: "Is there a list on the differences between pancake v3 and uniswap v3?" ([pancake-v3-contracts#24](https://github.com/pancakeswap/pancake-v3-contracts/issues/24))

### 12. DEX: token approvals, allowance, Permit2 and reverts

`execution reverted: STF` (insufficient allowance) is the classic Uniswap revert; Permit2 nonce and witness confusion is the modern one.

| Venues | Why it discriminates retrieval |
|---|---|
| uniswap, pancakeswap | Permit2 is deployed at the same canonical address on every chain for Uniswap (`0x000000000022D473030F116dDEE9F6B43aC78BA3`) but at a different address for PancakeSwap on the same BSC chain (`0x31c2F6fcFf4F8759b3Bd5Bf0e1084A055615c768`): same name, same chain, different answer (T2). |

Issue titles:

- "Calling exactInputSingle returns execution reverted: STF" ([Uniswap/docs#315](https://github.com/Uniswap/docs/issues/315))
- "AllowanceProvider failing to return `getAllowanceData` values on Sepolia" ([Uniswap/sdks#76](https://github.com/Uniswap/sdks/issues/76))

### 13. DEX: slippage tolerance, minimum-out and deadline

`slippageTolerance` semantics, `minimumAmountOut` arithmetic, and the fact that `SwapRouter02` dropped the `deadline` argument.

| Venues | Why it discriminates retrieval |
|---|---|
| uniswap, pancakeswap | The textbook error strings the brief suggested (`INSUFFICIENT_OUTPUT_AMOUNT`, `UniswapV2Router: EXPIRED`, `Transaction too old`) are in none of the issues quoted in this document. The corpus has `INSUFFICIENT_OUTPUT_AMOUNT` and `EXPIRED` in PancakeSwap's forms (its troubleshooting page uses `PancakeRouter: INSUFFICIENT_OUTPUT_AMOUNT` and `PancakeRouter: EXPIRED`) and has no chunk with `Transaction too old`, so they are adversarial material and are not presented as high-frequency developer questions. |

Issue titles:

<!-- verbatim GitHub issue title --><!-- vale crossvenue.Timeless = NO -->
- "`deadline` parameter is no longer exists in SwapRouter02 contract" ([Uniswap/sdks#234](https://github.com/Uniswap/sdks/issues/234))
<!-- vale crossvenue.Timeless = YES -->
- "Wrong minimumAmountOut calculation" ([v3-sdk#147](https://github.com/Uniswap/v3-sdk/issues/147))

### 14. Funding rate values, intervals and history

Funding is not universally 8-hourly; sign conventions, history pagination, and which endpoint is "the" rate.

| Venues | Why it discriminates retrieval |
|---|---|
| bybit, bingx, hyperliquid, dydx, ccxt; not binance (spot-only corpus, zero funding chunks) and not the AMM DEXes | The perp venues each state their own interval and sign convention. "How often does Binance pay funding?" has no answer in the corpus, and every neighboring venue supplies a confident-sounding wrong chunk, so the theme is the source of real unanswerables. |

Issue titles:

- "Help understanding funding rate values returned in two different APIs" ([hyperliquid-python-sdk#147](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/147))
- "Zero funding rate bug affects 62 of 226 markets (27%)" ([v4-clients#380](https://github.com/dydxprotocol/v4-clients/issues/380))

### 15. Hyperliquid specifics: agent/API wallet, L1-action signing, nonces

EIP-712 signing of L1 actions, agent wallet vs master account, `signatureChainId`, vault address, the nonce window.

| Venues | Why it discriminates retrieval |
|---|---|
| hyperliquid, with a direct collision against dydx (T12) | Both venues use the exact phrase "API wallet" for different mechanisms. |

Issue titles:

- "Is signatureChainId intentionally hardcoded to 0x66eee in sign_user_signed_action?" ([hyperliquid-python-sdk#254](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/254))
- "[hyperliquid] Concurrent signed actions reuse a nonce and get rejected as duplicates" ([ccxt#30348](https://github.com/ccxt/ccxt/issues/30348))

### 16. dYdX v4 specifics: `goodTilBlock`, account sequence, short-term vs long-term orders

Order expiry by block height, account sequence mismatches, and the two order lifetimes of dYdX v4.

| Venues | Why it discriminates retrieval |
|---|---|
| dydx only | A single-venue fact source, and the foil for `recvWindow` questions: dYdX has no such concept, and expiry is a block height. |

Issue titles:

- "Issues with GoodTilBlock" ([v4-clients#105](https://github.com/dydxprotocol/v4-clients/issues/105))
- "BroadcastError: … account sequence mismatch, expected 11000, got 10999" ([v4-clients#282](https://github.com/dydxprotocol/v4-clients/issues/282))

### 17. Symbol and market-id mapping

Unified symbol (`ETH/USDT:USDT`) vs raw exchange market id; contract vs spot symbols.

| Venues | Why it discriminates retrieval |
|---|---|
| ccxt primarily, every venue secondarily | It is a client-library problem: the corpus's answer is a ccxt chunk whichever venue is named (not counted per question). The set has no question on it, so how it separates retrieval methods is not measured. |

Issue titles:

- "BadSymbol: binanceusdm does not have market symbol ETHUSDT" ([ccxt#15877](https://github.com/ccxt/ccxt/issues/15877))

### 18. Market-buy-by-cost semantics

Spot market buys that take a quote amount rather than a base amount.

| Venues | Why it discriminates retrieval |
|---|---|
| ccxt, binance, bybit | Same reason as 17: it is a ccxt abstraction question, so every gold label would be a chunk of `wiki/Manual.md`. |

Issue titles:

- "Question: createMarketBuyOrderRequiresPrice usage?" ([ccxt#24915](https://github.com/ccxt/ccxt/issues/24915))

## The sharpest near-duplicate traps in this corpus

The adversarial category is built on them. Each is verified against `data/processed/chunks.jsonl`; the values quoted are the corpus's own.

| # | Trap | Venues | Shared wording | How the answers differ |
|---|---|---|---|---|
| T1 | Fee tiers | uniswap ↔ pancakeswap | "fee tiers available for V3 (concentrated liquidity) pools", "0.01% … 1%" | Uniswap v3: 0.01 / 0.05 / 0.30 / 1.00%. PancakeSwap v3 (BSC/Monad): 0.01 / 0.05 / 0.25 / 1%. PancakeSwap on Solana: 18 tiers. V2 base fee 0.30% vs 0.25%. The literal constant `3000` is a Uniswap tier and not a PancakeSwap one. |
| T2 | Permit2 address | uniswap ↔ pancakeswap, same chain | "Permit2", "gasless/batched approvals" | Uniswap: `0x0000…78BA3`, canonical on every chain including BNB. PancakeSwap on BSC: `0x31c2…c768`. Answering one with the other approves the wrong spender. |
| T3 | Router revert strings | uniswap v2 ↔ pancakeswap (and uniswap v2↔v3↔v4 internally) | `EXPIRED`, `K`, literally identical identifiers | Prefixes differ (`UniswapV2:` vs `PancakeRouter:`/`Pancake:`) and the documented remedies differ: PancakeSwap says set slippage ≥12% for tax tokens and hold 30% more tokens than the trade; Uniswap says use the `…SupportingFeeOnTransfer` router functions. Uniswap v3 uses mnemonics (`LOK`, `TLU`), v4 uses hex selectors (`0x5212cba1`). |
| T4 | `recvWindow` maximum | binance ↔ bybit ↔ bingx | "recvWindow … request validity window … default 5000" | Binance: max 60000. Bybit FIX: default 5000, max 60000. BingX: "Default is 5000; maximum is 5000", so the standard "raise recvWindow to fix clock drift" advice is wrong for BingX. dYdX and Hyperliquid have no such parameter. |
| T5 | Rate-limit breach code and ban length | binance ↔ bybit ↔ bingx | "too many requests", "rate limit exceeded", "back off" | Binance: 429, then 418 auto-ban scaling from 2 minutes to 3 days, per-IP. Bybit: 403 "access too frequent", a wait of at least 10 minutes, 600 req / 5 s per IP plus 10 rps read / 5 rps write per UID. BingX: 429, then 418, limits printed per endpoint as "N/s per UID; M/s per IP". Hyperliquid: 1200 weight/min per IP and an address-based limit of 1 request per 1 USDC traded. `403` means three different things across three venues. |
| T6 | Post-only | bybit ↔ bingx ↔ binance ↔ hyperliquid ↔ dydx | "post only", "maker only", "rejected if it would take liquidity" | Bybit/BingX: a `timeInForce` value spelled `PostOnly`. Binance: order type `LIMIT_MAKER`, with no TIF; sending a TIF where not required raises `-1114`. Hyperliquid: `tif = "Alo"`, title case, and it cancels rather than rejects. dYdX: a flag on short-term orders. No venue in this corpus uses `GTX`. |
| T7 | WebSocket keepalive | all five API venues | "ping", "pong", "heartbeat" | Binance: client messages capped at 5/s (ping frames count), connection dies at 24 h, 1024 streams. Bybit: client sends `{"op":"ping"}`. BingX: server sends the literal text `Ping`, client must reply `Pong`, everything GZIP-compressed (and that page is in Chinese). Hyperliquid: server closes if it hasn't sent anything in 60 s. dYdX: protocol-level control frames every 30 s, pong within 10 s. |
| T8 | Binance mainnet vs testnet vs demo (intra-venue) | binance ↔ binance | near-identical text: of the mainnet chunks of `rest-api.md` (167), `web-socket-api.md` (226) and `errors.md` (64), 105, 155 and 57 have a testnet occurrence identical once the word "testnet" is removed | Byte-identical pairs (433) are one chunk each, citing the mainnet page with the testnet page in `sources`. The pairs that differ only by the word "testnet" are two chunks, told apart only by `doc_path`/`heading_path`, which is the trap. A third environment, Demo Mode at `demo-api.binance.com`, is separate from the testnet: "Binance's test environment" has two mutually exclusive correct answers. |
| T9 | Order-book resync recipe | binance ↔ bybit | near-verbatim step lists (closest chunk pair between `bybit/docs/v5/websocket/public/full-ob.mdx` and `binance/testnet/web-socket-streams.md`: TF-IDF cosine 0.774, with the vectorizer parameters of [reproduce](../docs/how-to/reproduce.md#compare-pancakeswap-with-uniswap); what it was fitted on for this pair is not recorded) | Binance uses `U`/`u` first/final update IDs against a REST `lastUpdateId`; Bybit uses a per-topic `seq`/`u` with explicit `snapshot`/`delta` message types. |
| T10 | `limit` and pagination | binance ↔ bybit ↔ bingx ↔ hyperliquid | "`limit`", "`startTime`/`endTime`", "Default … Maximum …" | Binance klines: default 500, max 1000, and `depth` weight scales with limit. Bybit: opaque `nextPageCursor`, limit [1,1000]. BingX: `pageIndex`/`pageSize`, and the same `startTime` field is milliseconds on `inviteAccountList` and days on `commissionDataList`, `third/commissionDataList` and `partnerData`. Hyperliquid `l2Book`: hard 20 levels per side regardless of any limit. |
| T11 | Funding interval "8 hours" | hyperliquid ↔ bybit ↔ (binance gap) | "8 hour funding rate", "funding interval" | Hyperliquid: the formula is an 8-hour rate but funding is charged hourly at 1/8, capped 4%/hour, settled on the spot oracle price not the mark price, asset-independent. Bybit: "each symbol has a different funding interval", and the field is `fundingInterval` in minutes on `instruments-info` but `fundingIntervalHour` as a string on `tickers`. Binance: no funding content at all. |
| T12 | "API wallet" | hyperliquid ↔ dydx | the phrase "API Wallet", "generate a key pair", "sign on behalf of the account" | Hyperliquid: an EIP-712 `approveAgent` action with `signatureChainId`; orders are signed with the agent, and queries take the master address; querying by agent address returns empty. dYdX: Cosmos permissioned keys with per-key scoping, no EIP-712, expiry via `goodTilBlock`. |
| T13 | ccxt as a shadow venue | ccxt ↔ every venue | `createOrder`, `setSandboxMode`, `fetchFundingInterval` | 654 of the 1,436 ccxt chunks have a raw-venue chunk at TF-IDF cosine ≥ 0.3 (vectorizer parameters of [reproduce](../docs/how-to/reproduce.md#compare-pancakeswap-with-uniswap); what it was fitted on for this count is not recorded). "How do I place a post-only order on Bybit?" can legitimately retrieve `wiki/exchanges/bybit.md`, which is right for ccxt users and wrong for anyone calling REST directly. ccxt's own manual says "the unification of `timeInForce` is a work in progress". |

PancakeSwap's documentation is its own text, although PancakeSwap forks Uniswap's protocol. Their TF-IDF similarity is a result in [findings](../docs/results/findings.md#pancakeswaps-median-best-match-cosine-to-uniswap-is-0143). PancakeSwap's corpus is consumer-facing product and FAQ prose; Uniswap's is developer protocol reference. The hazard is conceptual duplication with different numbers. The shared vocabulary (pool, tick, fee tier, router, hooks, flash accounting, ERC-6909) is expected to confuse a dense retriever. An off-venue measurement on questions naming one of the two would test it.

The corpus's worst lexical duplication is intra-venue (T8); the counts are a result in [findings](../docs/results/findings.md#108-of-chunk-occurrences-are-byte-identical). Each group of byte-identical text is stored as one chunk ([`docs/explanation/design.md`](../docs/explanation/design.md#exact-deduplication)).

## Use in the question set

The questions use themes 1–16 and none of 17–18 (ccxt-abstraction questions, whose answers are all in one document). The category targets are decision 2's; the themes count toward them as follows:

- fact: themes 1, 2, 3, 5, 8, 9, 10, 14, 15, 16, spread across venues.
- comparison: explicitly two-venue questions drawn from T1, T4, T5, T6, T7, T10, T11, T12. They cover three information needs: numeric divergence under a shared term, the same name for a different mechanism, and a capability present on one venue and absent on the other.
- multihop: questions needing two chunks from the same venue (for example "what is the maximum `limit` on Bybit klines and how do I page past it").
- adversarial: T1–T13, across the three sub-shapes of §4.
- unanswerable: Binance futures funding and leverage (themes 7 and 14 on Binance, whose documentation in the corpus is spot only), dYdX v3, Bybit changelog history, plus wrong-venue premises drawn from T4, T6 and T12.

Every adversarial and comparison item records its confusable venue in `notes`. `per_query.csv` records the venue of every retrieved chunk; a tally of which wrong venue outranked the asked one is not reported ([methodology §4](methodology.md#4-adversarial-cases-for-retrieval)).

## References

- GitHub issue titles: the linked issues.
- Measurements over the corpus: [findings](../docs/results/findings.md#pancakeswaps-median-best-match-cosine-to-uniswap-is-0143) and [duplicate text](../docs/results/findings.md#108-of-chunk-occurrences-are-byte-identical).

### Not verified

- `pancakeswap/pancake-frontend` is not reachable through the GitHub API, so PancakeSwap issue frequencies are a lower bound.
