from Semi_sklearn.Transform.Transformer import Transformer
from googletrans import Translator
class Translate(Transformer):
    """
    A set of functions used to augment data.
    Supported languages are:
    Language Name	Code
    Afrikaans	af
    Albanian	sq
    Arabic	ar
    Azerbaijani	az
    Basque	eu
    Bengali	bn
    Belarusian	be
    Bulgarian	bg
    Catalan	ca
    Chinese Simplified	zh-CN
    Chinese Traditional	zh-TW
    Croatian	hr
    Czech	cs
    Danish	da
    Dutch	nl
    English	en
    Esperanto	eo
    Estonian	et
    Filipino	tl
    Finnish	fi
    French	fr
    Galician	gl
    Georgian	ka
    German	de
    Greek	el
    Gujarati	gu
    Haitian Creole	ht
    Hebrew	iw
    Hindi	hi
    Hungarian	hu
    Icelandic	is
    Indonesian	id
    Irish	ga
    Italian	it
    Japanese	ja
    Kannada	kn
    Korean	ko
    Latin	la
    Latvian	lv
    Lithuanian	lt
    Macedonian	mk
    Malay	ms
    Maltese	mt
    Norwegian	no
    Persian	fa
    Polish	pl
    Portuguese	pt
    Romanian	ro
    Russian	ru
    Serbian	sr
    Slovak	sk
    Slovenian	sl
    Spanish	es
    Swahili	sw
    Swedish	sv
    Tamil	ta
    Telugu	te
    Thai	th
    Turkish	tr
    Ukrainian	uk
    Urdu	ur
    Vietnamese	vi
    Welsh	cy
    Yiddish	yi
    """
    def __init__(self,sorce='zh-CN',target='zh-CN'):
        super().__init__()
        self.source=sorce
        self.target=target
        self.translator=Translator()
    def transform(self,X):
        X = self.translator.translate(X, dest=self.target, src=self.source).text
        return X
