#include <stdio.h>
#define max 15



void stack_init(void);

int stack[max+1];
int t , val ,i ;

void main()
{
    stack_init();

    for (i=1;i<=15;i++)
    {
          printf("\n Input: ",i);
          scanf("%d",&val);
          if (val > 0)
             push(val);


          else if(val < 0)
            {
              top(val);
              if (top(val)>0)
                printf("\n Expected output is: %d \n",top(val));
              else if (top(val)==0)
                printf("\n Expected output is: Error - Stack is empty.\n");
            }

           else if (val == 0)
              pop(val);
    }





}


void stack_init(void)
{
    t=-1;

}
int empty()
{

    if (t == -1)
        return 1;

    else
        return 0;

}
int push(int x)
{

    t=t+1;
    stack[t]=x;
}

int pop()
{
    int data;

    if(!empty())
        {
            data = stack[t];
            t = t - 1;
            return data;
        }
   else
    {
        printf("\n Expected output is: Error - Stack is empty.\n");
    }

}

int top()
{
    return stack[t];

}
