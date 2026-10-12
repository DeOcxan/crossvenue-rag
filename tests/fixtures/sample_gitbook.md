# Funding

Funding payments are exchanged between longs and shorts.&#x20;

{% hint style="info" %}
Funding is paid every hour, not every eight hours.
{% endhint %}

{% content-ref url="../trading/perpetuals.md" %}
[perpetuals.md](../trading/perpetuals.md)
{% endcontent-ref %}

## Funding rate formula

The premium is computed as `(mark - index) / index` and clamped to &lt;= 4%.

```python
premium = (mark_price - index_price) / index_price
funding_rate = clamp(premium, -0.04, 0.04)
```

{% tabs %}
{% tab title="REST" %}
Call `GET /info` with `type: "metaAndAssetCtxs"`.
{% endtab %}
{% endtabs %}

Previous
Next
