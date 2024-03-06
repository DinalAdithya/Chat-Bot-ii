class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'), // these selectors refers the claas name on the html
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }
    // to display messages
    display() {
        const {openButton, chatBox, sendButton} = this.args;

        //Button for chat bot window
        openButton.addEventListener('click', () => this.toggleState(chatBox))

        //Button for send msgs
        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }


    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');//extract the user input
        let text1 = textField.value
        if (text1 === "") { // check empty or not
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1); // push this object to msges array

        //http://127.0.0.1:5000/predict
        fetch($SCRIPT_ROOT+'/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())//extract the json responce
          .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);//pusing it to array again
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();