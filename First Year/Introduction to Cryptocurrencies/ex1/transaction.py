from .utils import PublicKey, TxID, Signature
from typing import Optional
import hashlib


class Transaction:
    """Represents a transaction that moves a single coin
    A transaction with no source creates money. It will only be created by the bank."""

    def __init__(self, output: PublicKey, input: Optional[TxID], signature: Signature) -> None:
        # do not change the name of this field:
        self._output: PublicKey = output
        # do not change the name of this field:
        self._input: Optional[TxID] = input
        # do not change the name of this field:
        self._signature: Signature = signature

    def get_txid(self) -> TxID:
        """Returns the identifier of this transaction. This is the SHA256 of the transaction contents."""
        # TODO: Ask Ofir if we need to add the signature to the hashing as well
        if self._input is not None:
            return hashlib.sha256(self._input + self._output).digest()
        return  hashlib.sha256(self._output).digest()