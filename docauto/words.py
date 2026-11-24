"""Word lists used by the built-in document types: names of taxes and tax ids worldwide, common currencies."""

# words that name a tax on documents worldwide (VAT, GST, sales tax...): used as labels of the tax amount
TAX_WORDS = ["VAT", "VAT Amount", "Total Tax", "GST", "HST", "PST", "Sales Tax", "IGST", "CGST", "SGST", "MwSt", "USt",
             "Mehrwertsteuer", "TVA", "IVA", "BTW", "Moms", "MVA", "ALV", "ΦΠΑ", "KDV", "PPN", "SST", "Consumption Tax",
             "消費税", "增值税", "税额", "부가세", "ضريبة القيمة المضافة", "الضريبة", "कर"]
# (not the bare word "Tax": it starts the "Tax Invoice" heading)
TAX_ID_WORDS = ["VAT No", "VAT Number", "VAT Reg No", "Tax ID", "Tax No", "TIN", "TRN", "GSTIN", "GST No", "ABN", "EIN",
                "USt-IdNr", "Steuernummer", "N° TVA", "Numéro de TVA", "SIRET", "NIF", "CIF", "P.IVA", "Partita IVA", "BTW-nr",
                "RFC", "CNPJ", "CUIT", "RUT", "NPWP", "UEN", "PAN", "الرقم الضريبي", "税号", "統一編號", "사업자등록번호"]
# the currencies most documents use; any other ISO 4217 code is kept too (allowOther)
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "CHF", "CAD", "AUD", "NZD", "SGD", "HKD", "AED", "SAR", "QAR", "KWD",
              "BHD", "OMR", "EGP", "ZAR", "NGN", "KES", "BRL", "MXN", "ARS", "CLP", "COP", "PEN", "SEK", "NOK", "DKK", "PLN",
              "CZK", "HUF", "RON", "TRY", "RUB", "ILS", "KRW", "TWD", "THB", "MYR", "IDR", "PHP", "VND", "PKR", "BDT", "LKR"]

# every ISO 4217 currency code in use (a currency printed next to an amount)
ISO_CURRENCIES = ("AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN BWP BYN BZD "
                  "CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD "
                  "GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT "
                  "LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR "
                  "NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP "
                  "STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XOF "
                  "XPF YER ZAR ZMW ZWL").split()
