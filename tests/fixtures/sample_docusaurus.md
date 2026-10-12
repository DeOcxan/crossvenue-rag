---
title: Create Order
sidebar_label: Create Order
description: Place a new order
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create Order

<!-- internal note: keep in sync with the v5 changelog -->

This endpoint places a new order. Covers spot, linear and inverse.

:::info
Only unified accounts can use `postOnly` together with `reduceOnly`.
:::

## Request Parameters

| Parameter | Required | Type | Comments |
|---|---|---|---|
| category | true | string | Product type: `spot`, `linear`, `inverse` |
| symbol | true | string | Symbol name, e.g. `BTCUSDT` |
| timeInForce | false | string | `GTC`, `IOC`, `FOK`, `PostOnly` |

Placeholders look like `POST /v5/order/create?symbol=<SYMBOL>` and must survive
normalisation intact.

## Request Example

<Tabs>
<TabItem value="http" label="HTTP">

```http
POST /v5/order/create HTTP/1.1
Host: api.bybit.com
X-BAPI-SIGN: XXXXX
Content-Type: application/json

{
    "category": "spot",
    "symbol": "BTCUSDT",
    "side": "Buy",
    "orderType": "Limit",
    "qty": "0.1",
    "price": "15600",
    "timeInForce": "PostOnly"
}
```

</TabItem>
</Tabs>

## Response Parameters

The `retCode` field is `0` on success. Any other value is an error and `retMsg`
carries the reason.

:::caution
Rate limit: 10 requests per second per UID.
:::
