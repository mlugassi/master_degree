from client.node import Node
from testing_utils import ONE_ETH, EthTools, RevertException
from client.utils import APPEAL_PERIOD, Contract, sign, ChannelStateMessage
import pytest


# here are tests for 3 basic scenarios. These also show how the work-flow with nodes proceeds.

def test_open_and_immediate_close(eth_tools: EthTools, alice: Node, bob: Node) -> None:
    eth_tools.start_tx_count()

    alice_init_balance = eth_tools.get_balance(alice.eth_address)
    bob_init_balance = eth_tools.get_balance(bob.eth_address)

    # Creating channel
    chan_address = alice.establish_channel(
        bob.eth_address, bob.ip_address, ONE_ETH)
    assert eth_tools.tx_count == 1

    # channel created, chan_address
    assert eth_tools.get_balance(chan_address) == ONE_ETH

    # ALICE CLOSING UNILATERALLY
    alice.close_channel(chan_address)
    assert eth_tools.tx_count == 2

    # Waiting
    eth_tools.mine_blocks(APPEAL_PERIOD+2)

    # Bob Withdraws (but this does not generate a transaction since his balance is 0)
    assert bob.withdraw_funds(chan_address) == 0

    # Alice Withdraws
    assert alice.withdraw_funds(chan_address) == ONE_ETH
    assert eth_tools.tx_count == 3

    assert eth_tools.get_balance(chan_address) == 0

    assert alice_init_balance == eth_tools.get_balance(
        alice.eth_address)
    assert bob_init_balance == eth_tools.get_balance(
        bob.eth_address)


def test_nice_open_transfer_and_close(eth_tools: EthTools,  alice: Node, bob: Node) -> None:
    alice_init_balance = eth_tools.get_balance(alice.eth_address)
    bob_init_balance = eth_tools.get_balance(bob.eth_address)

    # Creating channel
    chan_address = alice.establish_channel(
        bob.eth_address, bob.ip_address, 10*ONE_ETH)
    assert eth_tools.get_balance(chan_address) == 10*ONE_ETH

    # Alice sends money thrice
    eth_tools.start_tx_count()
    alice.send(chan_address, ONE_ETH)
    alice.send(chan_address, ONE_ETH)
    alice.send(chan_address, ONE_ETH)
    assert eth_tools.tx_count == 0

    # BOB CLOSING UNILATERALLY
    bob.close_channel(chan_address)

    # waiting
    eth_tools.mine_blocks(APPEAL_PERIOD+2)
    assert eth_tools.get_balance(chan_address) == 10*ONE_ETH

    # Bob Withdraws
    amount_withdrawn = bob.withdraw_funds(chan_address)
    assert amount_withdrawn == 3*ONE_ETH
    assert eth_tools.get_balance(chan_address) == 7*ONE_ETH

    # Alice Withdraws
    assert alice.withdraw_funds(chan_address) == 7*ONE_ETH
    assert eth_tools.get_balance(chan_address) == 0

    assert alice_init_balance == eth_tools.get_balance(
        alice.eth_address) + 3*ONE_ETH
    assert bob_init_balance == eth_tools.get_balance(
        bob.eth_address) - 3*ONE_ETH


def test_alice_tries_to_cheat(eth_tools: EthTools,  alice: Node, bob: Node) -> None:
    alice_init_balance = eth_tools.get_balance(alice.eth_address)
    bob_init_balance = eth_tools.get_balance(bob.eth_address)

    # Creating channel
    chan_address = alice.establish_channel(
        bob.eth_address, bob.ip_address, 10*ONE_ETH)

    # Alice sends money thrice
    alice.send(chan_address, ONE_ETH)
    old_state = alice.get_current_channel_state(chan_address)
    alice.send(chan_address, ONE_ETH)
    alice.send(chan_address, ONE_ETH)

    # ALICE TRIES TO CHEAT
    alice.close_channel(chan_address, old_state)

    # Waiting one block
    eth_tools.mine_blocks(1)

    # Bob checks if he needs to appeal, and sends an appeal
    assert bob.appeal_closed_chan(chan_address)

    # waiting
    eth_tools.mine_blocks(APPEAL_PERIOD)

    # Bob Withdraws
    assert bob.withdraw_funds(chan_address) == 3*ONE_ETH

    # Alice Withdraws
    assert alice.withdraw_funds(chan_address) == 7*ONE_ETH

    assert alice_init_balance == eth_tools.get_balance(
        alice.eth_address) + 3*ONE_ETH
    assert bob_init_balance == eth_tools.get_balance(
        bob.eth_address) - 3*ONE_ETH


def test_channel_list_encapsulation(alice: Node, chan: Contract) -> None:
    # We check that the node does not return an internal data structure which would allow
    # the user to modify the channel state without going through the API.
    chan_list = alice.get_list_of_channels()
    assert len(chan_list) == 1
    chan_list.clear()
    chan_list = alice.get_list_of_channels()
    assert len(chan_list) == 1


# a sample communication test between nodes
def test_node_rejects_receive_message_of_unknown_channel(eth_tools: EthTools, alice: Node, bob: Node, charlie: Node,
                                                         chan: Contract) -> None:
    eth_tools.start_tx_count()
    msg = ChannelStateMessage(
        chan.address, 5*ONE_ETH, 5*ONE_ETH, 10)
    signed_msg = sign(alice.private_key, msg)
    charlie.receive_funds(signed_msg)

    assert charlie.get_list_of_channels() == []
    with pytest.raises(Exception):
        charlie.get_current_channel_state(chan.address)
    assert eth_tools.tx_count == 0

# when we do something wrong, like close the contract twice
# we should be stopped both by the node and by the contract. Here we are stopped by the contract:


def test_close_by_alice_twice(alice: Node, chan: Contract, ) -> None:
    alice.send(chan.address, ONE_ETH)
    msg = alice.get_current_channel_state(chan.address)
    alice.close_channel(chan.address)
    v, r, s = msg.sig

    with pytest.raises(RevertException):
        chan.transact(alice, "oneSidedClose", (msg.balance1, msg.balance2, msg.serial_number,
                      v, r, s))


# Here the node refuses to close the closed channel once again (no transaction should be sent!)
def test_cant_close_channel_twice(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    alice.send(chan.address, ONE_ETH)
    alice.close_channel(chan.address)
    eth_tools.start_tx_count()
    with pytest.raises(Exception):
        alice.close_channel(chan.address)
    with pytest.raises(Exception):
        bob.close_channel(chan.address)
    assert eth_tools.tx_count == 0


def test_node_rejects_receive_message_of_unknown_channel(eth_tools: EthTools, alice: Node, bob: Node, charlie: Node,
                                                         chan: Contract) -> None:
    eth_tools.start_tx_count()
    msg = ChannelStateMessage(
        chan.address, 5*ONE_ETH, 5*ONE_ETH, 10)
    signed_msg = sign(alice.private_key, msg)
    charlie.receive_funds(signed_msg)

    assert charlie.get_list_of_channels() == []
    with pytest.raises(Exception):
        charlie.get_current_channel_state(chan.address)
    assert eth_tools.tx_count == 0
#######################################################################

def test_open_channel_again_after_close(eth_tools: EthTools, alice: Node, bob: Node) -> None:
    """בוחן פתיחת ערוץ מחדש אחרי סגירה"""
    alice_init_balance = eth_tools.get_balance(alice.eth_address)
    bob_init_balance = eth_tools.get_balance(bob.eth_address)

    # יצירת ערוץ
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, ONE_ETH)
    
    # סגירת הערוץ
    alice.close_channel(chan_address)

    # פתיחת ערוץ חדש
    new_chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, ONE_ETH)

    # וידוא שהערוץ החדש נפתח בצורה תקינה
    assert eth_tools.get_balance(new_chan_address) == ONE_ETH

    # בדיקת יתרה לפני ואחרי
    # assert alice_init_balance == eth_tools.get_balance(alice.eth_address)
    assert bob_init_balance == eth_tools.get_balance(bob.eth_address)

def test_send_different_amounts(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בודק שליחה של סכומים שונים בתוך הערוץ"""
    alice_init_balance = eth_tools.get_balance(alice.eth_address)
    bob_init_balance = eth_tools.get_balance(bob.eth_address)

    amounts = [ONE_ETH, Web3.to_wei(2, 'ether'), Web3.to_wei(0.5, 'ether')]

    for amount in amounts:
        alice.send(chan.address, amount)
        eth_tools.mine_blocks(1)

    # וידוא שהיתרה של הערוץ השתנתה כמצופה
    assert eth_tools.get_balance(chan.address) == 10 * ONE_ETH - sum(amounts)

    # בדוק את היתרה של אליס ובוב
    assert alice_init_balance == eth_tools.get_balance(alice.eth_address) + sum(amounts)
    assert bob_init_balance == eth_tools.get_balance(bob.eth_address)

def test_appeal_period(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בודק שהערעור על סגירת ערוץ במהלך תקופת הערעור עובד כראוי"""
    eth_tools.start_tx_count()

    # יצירת ערוץ
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, 5*ONE_ETH)
    
    # אליס שולחת כסף
    alice.send(chan.address, ONE_ETH)
    
    # בוב סוגר את הערוץ באופן חד צדדי
    bob.close_channel(chan.address)
    
    # חכים עד שהתקופה תסיים
    eth_tools.mine_blocks(APPEAL_PERIOD)
    
    # בוב בודק אם יש צורך להגיש ערעור
    assert bob.appeal_closed_chan(chan.address)
    
    # בוב מושך את כספו
    assert bob.withdraw_funds(chan.address) == 1*ONE_ETH
    assert alice.withdraw_funds(chan.address) == 4*ONE_ETH

def test_invalid_signature(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בודק אם המערכת מזהה חתימה לא תקינה"""
    alice.send(chan.address, ONE_ETH)
    msg = alice.get_current_channel_state(chan.address)
    
    # שינוי לא תקין בחתימה
    invalid_signature = (msg.serial_number, msg.balance1, msg.balance2)
    
    with pytest.raises(RevertException):
        chan.transact(alice, "oneSidedClose", invalid_signature)

def test_cancel_close_after_single_close(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בודק אם ניתן לבטל את הסגירה של הערוץ לאחר סגירה חד צדדית"""
    eth_tools.start_tx_count()

    # יצירת ערוץ
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, 5*ONE_ETH)

    # אליס שולחת כסף
    alice.send(chan.address, ONE_ETH)

    # סגירה חד צדדית של אליס
    alice.close_channel(chan.address)

    # וידוא שהערוץ נסגר ושתקופת הערעור מתחילה
    eth_tools.mine_blocks(APPEAL_PERIOD+2)
    
    with pytest.raises(Exception):
        # אליס לא יכולה לסגור את הערוץ פעם נוספת
        alice.close_channel(chan.address)

def test_simple_transaction(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בדיקה פשוטה של שליחת עסקה בין שני צדדים"""
    init_balance_alice = eth_tools.get_balance(alice.eth_address)
    init_balance_bob = eth_tools.get_balance(bob.eth_address)
    
    # אליס שולחת עסקה לבוב
    amount = Web3.to_wei(1, 'ether')
    alice.send(chan.address, amount)
    eth_tools.mine_blocks(1)

    # בדוק את יתרת המשתמשים
    assert eth_tools.get_balance(alice.eth_address) == init_balance_alice - amount
    assert eth_tools.get_balance(bob.eth_address) == init_balance_bob + amount

def test_single_party_close_with_appeal(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בדיקת סגירת ערוץ חד-צדדית עם אפשרות להגיש ערעור"""
    # יצירת ערוץ ושליחת סכום
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, 5 * ONE_ETH)
    alice.send(chan.address, Web3.to_wei(2, 'ether'))
    
    # אליס סוגרת את הערוץ חד-צדדית
    alice.close_channel(chan.address)
    
    # אורך תקופת הערעור
    eth_tools.mine_blocks(APPEAL_PERIOD)

    # בדוק אם ניתן להגיש ערעור
    assert alice.submit_appeal(chan_address) == True

def test_channel_balance_after_close(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בדיקה של יתרת הערוץ לאחר סגירה"""
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, 3 * ONE_ETH)
    
    # שליחה של סכום ערוץ
    alice.send(chan.address, Web3.to_wei(1, 'ether'))
    
    # סגירת הערוץ
    alice.close_channel(chan_address)
    
    # וידוא שהיתרה של הערוץ מתעדכנת
    assert eth_tools.get_balance(chan_address) == 2 * ONE_ETH

def test_appeal_period_functionality(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בדיקה של תקופת הערעור אחרי סגירת ערוץ"""
    # יצירת ערוץ
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, 5 * ONE_ETH)
    
    # סגירה חד-צדדית
    alice.close_channel(chan_address)
    
    # חפירת בלוקים ותחילת תקופת הערעור
    eth_tools.mine_blocks(APPEAL_PERIOD)

    # הגשה של ערעור על סגירת הערוץ
    assert alice.submit_appeal(chan_address) == True

def test_invalid_signature_submission(eth_tools: EthTools, alice: Node, bob: Node, chan: Contract) -> None:
    """בדיקת חתימה לא תקינה בעת הגשת ערעור"""
    chan_address = alice.establish_channel(bob.eth_address, bob.ip_address, ONE_ETH)
    
    # סגירת הערוץ
    alice.close_channel(chan_address)
    
    # ניסיון לשלוח חתימה לא תקינה
    invalid_signature = "invalid_signature_data"
    with pytest.raises(RevertException):
        alice.submit_appeal(chan_address, signature=invalid_signature)
